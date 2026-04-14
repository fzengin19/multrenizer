# Multrenizer

Multrenizer is a bilingual English-Turkish Unigram tokenizer built from scratch for Turkish morphology, Turkish-aware casing, and mixed TR-EN text.

## Links

- Repository: [github.com/fzengin19/multrenizer](https://github.com/fzengin19/multrenizer)

## Why Multrenizer?

Standard multilingual tokenizers routinely break Turkish at poor boundaries, waste context on agglutinative suffixes, and mishandle the Turkish dotted/dotless `I/i` rule. Multrenizer is designed to fix those failure modes without discarding punctuation and chat-critical symbols.

Core design goals:

- Turkish-aware normalization: hardcoded `İ -> i` and `I -> ı` before Unicode normalization
- Apostrophe preservation: forms like `feature'ı`, `merge'lemek`, `İstanbul'da`, and `can't` keep `'` as a real token
- Compact vocabulary budget: `~26K` target vocab for a Turkish-first bilingual tokenizer
- Fixed utility budget: dedicated punctuation, emoji, math, currency, and chat symbols
- Code-switching support: trained on mixed TR-EN text instead of treating it as noise

## Benchmark Results

Evaluated on `5,000` Turkish sentences, `5,000` English sentences, and `500` code-switching sentences from the prepared corpus against 5 reference tokenizers.

Notes:

- Multrenizer's shipped local artifact is auto-read from `multrenizer-tokenizer/tokenizer.json`; the current released artifact is `25,917` tokens.
- Example token strings for byte-level models are shown as raw tokenizer pieces. Metrics are based on exact token counts, not prettified decoding.

### Compared Tokenizers

| Tokenizer | Source | Vocab Size | Algorithm | Type |
|---|---|---:|---|---|
| **Multrenizer** | This project | **25,917** | Unigram | Bilingual EN-TR, purpose-built |
| **Kumru-2B** | [vngrs-ai/Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) | 50,176 | BPE | Turkish LLM (VNGRS, Sep 2025, Mistral-based) |
| **Turkcell-7B** | [TURKCELL/Turkcell-LLM-7b-v1](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1) | 48,351 | BPE | Turkish LLM (Turkcell, Apr 2024, Mistral-based) |
| **GPT-2** | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 50,257 | BPE | English-centric baseline (OpenAI, 2019) |
| **Qwen-3** | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 151,643 | BPE | Multilingual (Alibaba, 2025) |
| **Mistral-3.1** | [mistralai/Mistral-Small-3.1-24B-Base-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503) | 131,072 | BPE/SP | Multilingual (Mistral AI, Mar 2025) |

### Fertility, Compression, and Token Count

Lower fertility means fewer tokens per word. Higher compression means more characters carried per token.

| Metric | Multrenizer | Kumru-2B | Turkcell-7B | GPT-2 | Qwen-3 | Mistral-3.1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vocab Size | **25,917** | 50,176 | 48,351 | 50,257 | 151,643 | 131,072 |
| **TR Fertility** | **1.627** | 1.649 | 1.917 | 3.785 | 2.616 | 2.384 |
| EN Fertility | 1.525 | 2.151 | 1.555 | **1.314** | 1.372 | 1.381 |
| **CS Fertility** | **1.756** | 1.923 | 1.832 | 3.475 | 2.445 | 2.479 |
| **TR Compression** | **4.783** | 4.719 | 4.060 | 2.056 | 2.976 | 3.265 |
| EN Compression | 4.148 | 2.942 | 4.068 | **4.816** | 4.610 | 4.580 |
| **TR Total Tokens (5K)** | **130,844** | 132,637 | 154,166 | 304,345 | 210,334 | 191,682 |
| EN Total Tokens (5K) | 157,027 | 221,420 | 160,121 | **135,235** | 141,275 | 142,196 |
| **CS Total Tokens (500)** | **5,525** | 6,050 | 5,762 | 10,933 | 7,693 | 7,799 |

Current position:

- Best Turkish efficiency in this comparison set: TR fertility, TR compression, TR total tokens
- Best code-switching efficiency in this comparison set: CS fertility and CS total tokens
- Competitive English coverage for a Turkish-first tokenizer, but not better than English-native GPT-2 on EN-only token count
- Only tokenizer here that passes Turkish `I/i` normalization correctly

### Morphological Splitting

Total tokens needed to represent 10 difficult Turkish words:

| Tokenizer | Vocab Size | Total Tokens | Avg per Word |
|---|---:|:---:|:---:|
| **Multrenizer** | **25,917** | **32** | **3.2** |
| Kumru-2B | 50,176 | 35 | 3.5 |
| Turkcell-7B | 48,351 | 38 | 3.8 |
| Mistral-3.1 | 131,072 | 71 | 7.1 |
| Qwen-3 | 151,643 | 73 | 7.3 |
| GPT-2 | 50,257 | 105 | 10.5 |

Selected examples:

```text
güzelleştirilmiş
  Multrenizer: güzel + leştirilmiş                                   [2 tokens]
  Kumru-2B: 2 tokens
  Turkcell-7B: güzel + leştirilmiş                                   [2 tokens]
  Qwen-3: 5 tokens
  Mistral-3.1: 5 tokens
  GPT-2: 10 tokens

İstanbul'da
  Multrenizer: istanbul + ' + da                                     [3 tokens]
  Kumru-2B: 3 tokens
  Turkcell-7B: İstanbul + ' + da                                     [3 tokens]
  Qwen-3: 4 tokens
  Mistral-3.1: 4 tokens
  GPT-2: 5 tokens

Afyonkarahisarlılaştıramadıklarımızdan
  Multrenizer: afyonkarahisar + lı + laştı + r + ama + dıkları + mızda + n   [8 tokens]
  Kumru-2B: 8 tokens
  Turkcell-7B: 9 tokens
  Qwen-3: 16 tokens
  Mistral-3.1: 16 tokens
  GPT-2: 21 tokens
```

### Turkish I/i Normalization

This is the critical locale-sensitive test:

- `İ` must lowercase to `i`
- `I` must lowercase to `ı`

| Input | Expected | Multrenizer | Kumru-2B | Turkcell-7B | GPT-2 | Qwen-3 | Mistral-3.1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| İstanbul | istanbul | **OK** | FAIL | FAIL | FAIL | FAIL | FAIL |
| IŞIK | ışık | **OK** | FAIL | FAIL | FAIL | FAIL | FAIL |
| SIR | sır | **OK** | FAIL | FAIL | FAIL | FAIL | FAIL |
| İNSAN | insan | **OK** | FAIL | FAIL | FAIL | FAIL | FAIL |
| ISITMAK | ısıtmak | **OK** | FAIL | FAIL | FAIL | FAIL | FAIL |
| **Score** | | **8/8** | **0/8** | **0/8** | **0/8** | **0/8** | **0/8** |

Multrenizer is the only tokenizer in this comparison that handles Turkish casing correctly.

### Code-Switching

```text
"Bu feature'ı implement ederken edge case'leri handle etmeyi unutmayalım."

  Multrenizer  [15 tok]  bu | feature | ' | ı | implement | ederken | edge | case | ' | leri | handle | etmeyi | unutmaya | lım | .
  Kumru-2B     [20 tok]  Bu | fe | ature | ' | Ä± | imp | lement | ederken | ed | ge | cas | e | ' | leri | hand | le | etmeyi | unutma | yalÄ±m | .
  Turkcell-7B  [15 tok]  Bu | feature | ' | ı | implement | ederken | edge | case | ' | leri | handle | etmeyi | unut | mayalım | .
  GPT-2        [24 tok]  Bu | feature | ' | Ä± | implement | ed | er | ken | edge | case | ' | ler | i | handle | et | me | yi | un | ut | may | al | Ä± | m | .
  Qwen-3       [22 tok]  Bu | feature | ' | Ä± | implement | ed | er | ken | edge | case | ' | leri | handle | et | m | ey | i | un | ut | may | alÄ±m | .
  Mistral-3.1  [20 tok]  Bu | feature | 'Ä± | implement | eder | ken | edge | case | ' | leri | handle | et | me | yi | un | ut | may | al | Ä±m | .

"merge'lemek istediğim branch conflict veriyor."

  Multrenizer  [ 8 tok]  merge | ' | lemek | istediğim | branch | conflict | veriyor | .
  Kumru-2B     [14 tok]  mer | ge | ' | lemek | istediÄŁim | b | ran | ch | con | f | lic | t | veriyor | .
  Turkcell-7B  [ 8 tok]  merge | ' | lemek | istediğim | branch | conflict | veriyor | .
  GPT-2        [16 tok]  mer | ge | ' | lem | ek | is | ted | i | ÄŁ | im | branch | conflict | ver | iy | or | .
  Qwen-3       [11 tok]  merge | ' | lem | ek | istediÄŁ | im | branch | conflict | ver | iyor | .
  Mistral-3.1  [13 tok]  merge | ' | le | mek | ist | edi | ÄŁ | im | branch | conflict | ver | iyor | .
```

## Quick Start

### Installation

```bash
git clone https://github.com/fzengin19/multrenizer.git
cd multrenizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Use the shipped tokenizer

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("multrenizer-tokenizer/tokenizer.json")

encoded = tok.encode("İstanbul'da güzel bir gün")
print(encoded.tokens)
# ['<s>', 'istanbul', "'", 'da', 'güzel', 'bir', 'gün', '</s>']

print(tok.normalizer.normalize_str("IŞIK"))
# 'ışık'
```

### Train from scratch

```bash
# 1. Download and prepare corpus
python prepare_data.py --size medium

# 2. Train tokenizer
python train_tokenizer.py --data-dir data/

# 3. Optional: push tokenizer files to Hugging Face Hub
python train_tokenizer.py --data-dir data/ \
  --repo-id your-username/multrenizer \
  --hf-token hf_xxxxx
```

### Run benchmarks

```bash
python benchmark.py --tr-lines 5000 --en-lines 5000
```

## Architecture

### Pipeline

```text
Raw text
  -> Turkish I/i normalizer (Replace: İ->i, I->ı, i̇->i)
  -> Quote canonicalization (’ ‘ ʼ ＇ -> ')
  -> NFKC normalization
  -> Lowercase
  -> Strip whitespace
  -> Pre-tokenizer (whitespace + apostrophe + punctuation split)
  -> Unigram model (~26K target vocab)
  -> Post-processor (<s> ... </s>)
```

### Data Mix

The released artifact is trained with the default file-based interleave in `train_tokenizer.py`, which approximates:

| Stream | Share | Purpose |
|---|---|---|
| Turkish | ~60% | Core Turkish morphology |
| English | ~30% | English coverage |
| Code-switching | ~10% | TR-EN boundary handling |

Corpus collection is Turkish-forward, and code-switching examples are generated from OPUS parallel pairs during data preparation.

### Vocabulary Budget

Multrenizer is designed around a `26,000` target vocabulary, with a fixed budget reserved for always-preserved tokens:

- `32` named special tokens
- `512` reserved tokens
- `292` utility tokens
- up to `25,164` learned subword tokens

Current shipped artifact: `25,917` total tokens.

### Special Tokens

| Category | IDs | Tokens | Purpose |
|---|---|---|---|
| **Core** | 0-3 | `<unk>` `<s>` `</s>` `<pad>` | Basic tokenizer operation |
| **Chat** | 4-8 | `<\|system\|>` `<\|user\|>` `<\|assistant\|>` `<\|end\|>` `<\|sep\|>` | Instruction tuning and chat models |
| **Reasoning** | 9-12 | `<think>` `</think>` `<\|step\|>` `<\|reflection\|>` | Reasoning traces and self-check markers |
| **Tool Use** | 13-16 | `<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>` | Tool and function calling |
| **Code/FIM** | 17-20 | `<\|code\|>` `<\|fim_prefix\|>` `<\|fim_middle\|>` `<\|fim_suffix\|>` | Code and fill-in-middle workflows |
| **Bilingual** | 21-22 | `<\|tr\|>` `<\|en\|>` | Language tags |
| **RAG** | 23-24 | `<\|context\|>` `<\|/context\|>` | Retrieval boundaries |
| **Multi-modal** | 25-28 | `<\|image\|>` `<\|audio\|>` `<\|video\|>` `<\|file\|>` | Placeholder tokens |
| **Structured** | 29-31 | `<\|json\|>` `<\|table\|>` `<\|cite\|>` | Structured output markers |
| **Reserved** | 32-543 | `<\|reserved_0\|>` ... `<\|reserved_511\|>` | Future growth without retraining |
| **Utility** | 544+ | Punctuation, emoji, math, currency, typography | Critical text symbols kept intact |

### Utility Tokens

| Category | Count | Examples |
|---|---:|---|
| Punctuation | 31 | `. , ! ? ; : - ( ) [ ] { } / \ " ' ...` |
| Currency & Business | 15 | `₺ $ € £ ¥ % @ # &` |
| Math & Science | 25 | `± × ÷ ≠ ≤ ≥ ∞ √ π α β γ` |
| Arrows & Symbols | 15 | `→ ← ↑ ↓ • ★ ☆ ✓ ✗ © ® ™` |
| Typography | 10 | `« » “ ” ‘ ’ ‹ › „ ‚` |
| Emoji (faces) | 70 | `😀 😂 🤣 😊 😍 🤔 😭 😡 💀 🤖` |
| Emoji (hands) | 28 | `👋 👍 👎 👏 🙏 💪 ✊ ✌️` |
| Emoji (hearts) | 18 | `❤️ 💛 💚 💙 💜 🖤 💔` |
| Emoji (symbols) | 36 | `🔥 ✨ ⭐ ✅ ❌ ⚠️ 💯 🚀` |
| Emoji (objects) | 36 | `💻 📱 🎯 🏆 📊 ☕ 🔗 💰` |
| Emoji (flags) | 8 | `🇹🇷 🇺🇸 🇬🇧 🇩🇪 🇫🇷 🇪🇸 🇮🇹 🇯🇵` |

## Project Structure

```text
multrenizer/
├── multrenizer-tokenizer/     # Trained tokenizer artifact
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── prepare_data.py            # Corpus download and preparation
├── train_tokenizer.py         # Tokenizer training script
├── benchmark.py               # Benchmark against 5 reference tokenizers
├── benchmark_results.json     # Full benchmark output
├── tests/                     # Regression tests for tokenizer behavior
├── requirements.txt
└── pyproject.toml
```

## References

- [Tokens with Meaning: A Hybrid Tokenization Approach for Turkish](https://arxiv.org/html/2508.14292v2)
- [Tokenization Standards for Linguistic Integrity: Turkish as a Benchmark](https://arxiv.org/html/2502.07057v1)
- [Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE](https://arxiv.org/abs/2508.08424)
- [Vocabulary Trimming: An Easy and Effective Method for SLM Acceleration](https://blog.squeezebits.com/vocabulary-trimming-methods)

## License

Apache 2.0
