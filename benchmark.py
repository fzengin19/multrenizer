#!/usr/bin/env python3
"""
Multrenizer - Large-Scale Benchmark & Comparison
==================================================
Compares Multrenizer against 5 major tokenizers using 5,000+ real sentences
per language from the prepared corpus.

Tokenizers:
  - Multrenizer        : ~26K target vocab, Unigram, bilingual EN-TR (ours)
  - Kumru-2B           :  50,176 vocab, BPE, Turkish LLM tokenizer (VNGRS, Sep 2025)
  - Turkcell-LLM-7B    :  48,351 vocab, BPE, Turkish LLM tokenizer (Turkcell, Apr 2024)
  - GPT-2              :  50,257 vocab, BPE, English-centric (OpenAI)
  - Qwen-3             : 151,643 vocab, BPE, multilingual (Alibaba, 2025)
  - Mistral-3.1        : 131,072 vocab, BPE/SP, multilingual (Mistral AI, Mar 2025)

Metrics:
  1. Fertility (tokens per word) on 5K+ real sentences per language
  2. Compression ratio (characters per token)
  3. Turkish morphological splitting quality (10 test words)
  4. Turkish I/i normalization correctness (8 test cases)
  5. Code-switching tokenization examples

Notes:
  - The local Multrenizer vocab size is auto-read from tokenizer.json.
  - Displayed token strings are raw tokenizer pieces with only boundary markers removed.
    Byte-level BPE tokenizers may therefore show UTF-8 fragments such as "Ä±".
"""

import argparse
import json
import os
import random
import sys
import time

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"
RESULTS_PATH = "benchmark_results.json"

TOKENIZER_REGISTRY = {
    "Multrenizer": {
        "type": "local",
        "path": "./multrenizer-tokenizer",
        "vocab": 26_000,
        "algo": "Unigram",
        "desc": "Bilingual EN-TR, Turkish-aware normalization",
    },
    "Kumru-2B": {
        "type": "hub",
        "model_id": "vngrs-ai/Kumru-2B",
        "vocab": 50_176,
        "algo": "BPE",
        "desc": "Turkish LLM tokenizer (VNGRS, Sep 2025, Mistral-based)",
    },
    "Turkcell-7B": {
        "type": "hub",
        "model_id": "TURKCELL/Turkcell-LLM-7b-v1",
        "vocab": 48_351,
        "algo": "BPE",
        "desc": "Turkish LLM tokenizer (Turkcell, Apr 2024, Mistral-based)",
    },
    "GPT-2": {
        "type": "hub",
        "model_id": "openai-community/gpt2",
        "vocab": 50_257,
        "algo": "BPE",
        "desc": "English-centric baseline (OpenAI, 2019)",
    },
    "Qwen-3": {
        "type": "hub",
        "model_id": "Qwen/Qwen3-0.6B",
        "vocab": 151_643,
        "algo": "BPE",
        "desc": "Multilingual (Alibaba, 2025)",
    },
    "Mistral-3.1": {
        "type": "hub",
        "model_id": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "vocab": 131_072,
        "algo": "BPE/SentencePiece",
        "desc": "Multilingual (Mistral AI, Mar 2025)",
    },
}

# Short names for table display (max 12 chars)
SHORT_NAMES = {
    "Multrenizer": "Multrenizer",
    "Kumru-2B": "Kumru-2B",
    "Turkcell-7B": "Turkcell-7B",
    "GPT-2": "GPT-2",
    "Qwen-3": "Qwen-3",
    "Mistral-3.1": "Mistral-3.1",
}

# Morphological test words
TR_MORPHOLOGY_WORDS = [
    ("güzelleştirilmiş", "güzel + leştir + il + miş"),
    ("evlerimizdekilerin", "ev + ler + imiz + deki + ler + in"),
    ("çalışmalarımızın", "çalış + ma + lar + ımız + ın"),
    ("okuduklarımdan", "oku + duk + lar + ım + dan"),
    ("Afyonkarahisarlılaştıramadıklarımızdan",
     "Afyonkarahisar + lı + laş + tır + ama + dık + lar + ımız + dan"),
    ("İstanbul'da", "İstanbul + da"),
    ("değerlendirememişlerdi", "değer + lendir + eme + miş + ler + di"),
    ("kullanılabilirlik", "kullan + ıl + abil + ir + lik"),
    ("özelleştirilebilecek", "özel + leştir + ile + bil + ecek"),
    ("düşünülebileceğini", "düşün + üle + bil + eceğ + in + i"),
]

# Turkish I/i normalization test cases
TURKISH_I_CASES = [
    ("İstanbul", "istanbul", "İ -> i"),
    ("IŞIK", "ışık", "I -> ı"),
    ("İŞ", "iş", "İ -> i"),
    ("SIR", "sır", "I -> ı (not 'sir')"),
    ("İLK", "ilk", "İ -> i"),
    ("IRAK", "ırak", "I -> ı"),
    ("İNSAN", "insan", "İ -> i"),
    ("ISITMAK", "ısıtmak", "I -> ı"),
]

# Code-switching sentences
CS_SENTENCES = [
    "Deploy sürecinde unexpected bir error aldık, rollback yapmamız gerekiyor.",
    "Bu feature'ı implement ederken edge case'leri handle etmeyi unutmayalım.",
    "Meeting'i reschedule edelim, stakeholder'lar available değil bugün.",
    "Code review'da feedback aldım, design pattern'ı değiştirmemiz lazım.",
    "merge'lemek istediğim branch conflict veriyor.",
    "Sprint planning'de discuss ettiğimiz gibi, backend refactoring'i önceliklendirmeliyiz.",
    "Production'a push etmeden önce staging'de test edelim.",
    "Database migration'ı run ettikten sonra cache'i invalidate etmeyi unutma.",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus_lines(filepath: str, max_lines: int) -> list[str]:
    """Load up to max_lines from a corpus file."""
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and len(stripped) >= 20:
                lines.append(stripped)
                if len(lines) >= max_lines:
                    break
    return lines


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_all_tokenizers() -> dict:
    """Load all tokenizers from registry. Returns {name: (hf_tokenizer, raw_tokenizer_or_None)}."""
    loaded = {}
    for name, cfg in TOKENIZER_REGISTRY.items():
        print(f"  Loading {name}...", end=" ", flush=True)
        try:
            if cfg["type"] == "local":
                raw = Tokenizer.from_file(os.path.join(cfg["path"], "tokenizer.json"))
                cfg["vocab"] = raw.get_vocab_size()
                tok = PreTrainedTokenizerFast(
                    tokenizer_object=raw,
                    bos_token="<s>", eos_token="</s>",
                    unk_token="<unk>", pad_token="<pad>",
                )
                loaded[name] = (tok, raw)
            else:
                tok = AutoTokenizer.from_pretrained(cfg["model_id"])
                loaded[name] = (tok, None)
            print(f"OK (vocab={cfg['vocab']:,})")
        except Exception as e:
            print(f"FAILED: {e}")
    return loaded


def get_vocab_size_for_display(name: str, all_tok: dict) -> int:
    """Use the shipped local artifact's true vocab size when available."""
    _, raw = all_tok[name]
    if raw is not None:
        return raw.get_vocab_size()
    return TOKENIZER_REGISTRY[name]["vocab"]


# ---------------------------------------------------------------------------
# Metrics on large data
# ---------------------------------------------------------------------------

def calc_fertility_bulk(tokenizer, sentences: list[str]) -> float:
    """Fertility = total_tokens / total_words over all sentences."""
    total_tokens = 0
    total_words = 0
    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        words = sent.split()
        total_tokens += len(tokens)
        total_words += len(words)
    return total_tokens / total_words if total_words else 0


def encode_without_specials(tokenizer, text: str) -> list[int]:
    """Encode text without BOS/EOS or model-specific special tokens."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def clean_display_token(token: str) -> str:
    """Remove tokenizer-specific boundary markers without altering raw bytes."""
    return token.replace("▁", "").replace("Ġ", "").replace("##", "")


def calc_compression_bulk(tokenizer, sentences: list[str]) -> float:
    """Compression = total_chars / total_tokens over all sentences."""
    total_chars = 0
    total_tokens = 0
    for sent in sentences:
        ids = encode_without_specials(tokenizer, sent)
        total_chars += len(sent)
        total_tokens += len(ids)
    return total_chars / total_tokens if total_tokens else 0


def calc_total_tokens(tokenizer, sentences: list[str]) -> int:
    """Total token count for a list of sentences."""
    return sum(len(encode_without_specials(tokenizer, s)) for s in sentences)


def tokenize_word(tokenizer, word: str) -> list[str]:
    """Tokenize a single word, preserving raw tokenizer output for comparison."""
    tokens = tokenizer.tokenize(word)
    cleaned = [clean_display_token(t) for t in tokens]
    return [t for t in cleaned if t] if cleaned else tokens


def check_normalization(raw_tokenizer, word: str) -> str:
    """Check normalization via raw tokenizer (Multrenizer) or fallback to .lower()."""
    if raw_tokenizer and hasattr(raw_tokenizer, "normalizer") and raw_tokenizer.normalizer:
        return raw_tokenizer.normalizer.normalize_str(word)
    return word.lower()


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------

def fmt_table_row(label: str, values: dict, names: list[str], fmt: str = ".2f") -> str:
    """Format a table row with label + values for each tokenizer."""
    cells = f"  {label:<26}"
    for n in names:
        val = values.get(n, 0)
        if isinstance(val, float):
            cells += f" {val:>12{fmt}}"
        elif isinstance(val, int):
            cells += f" {val:>12,}"
        else:
            cells += f" {str(val):>12}"
    return cells


def print_table_header(names: list[str]):
    """Print table header with tokenizer names."""
    header = f"  {'Metric':<26}"
    for n in names:
        header += f" {SHORT_NAMES.get(n, n):>12}"
    print(header)
    print("  " + "-" * (26 + 13 * len(names)))


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(tr_lines: int = 5000, en_lines: int = 5000, cs_lines: int = 500):
    names_order = list(TOKENIZER_REGISTRY.keys())

    print("=" * 90)
    print("  MULTRENIZER LARGE-SCALE BENCHMARK")
    print("  6 Tokenizers | 5K+ Real Sentences per Language")
    print("=" * 90)

    # --- Load data ---
    print("\n[1/6] Loading corpus data...")
    tr_path = os.path.join(DATA_DIR, "tr_corpus.txt")
    en_path = os.path.join(DATA_DIR, "en_corpus.txt")
    cs_path = os.path.join(DATA_DIR, "cs_corpus.txt")

    for p in [tr_path, en_path]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found. Run prepare_data.py first.")
            sys.exit(1)

    tr_data = load_corpus_lines(tr_path, tr_lines)
    en_data = load_corpus_lines(en_path, en_lines)
    cs_data = load_corpus_lines(cs_path, cs_lines) if os.path.exists(cs_path) else CS_SENTENCES

    print(f"  TR sentences: {len(tr_data):,}")
    print(f"  EN sentences: {len(en_data):,}")
    print(f"  CS sentences: {len(cs_data):,}")

    # --- Load tokenizers ---
    print("\n[2/6] Loading tokenizers...")
    all_tok = load_all_tokenizers()
    names_order = [n for n in names_order if n in all_tok]

    results = {"config": {
        "tr_sentences": len(tr_data),
        "en_sentences": len(en_data),
        "cs_sentences": len(cs_data),
        "tokenizers": {n: TOKENIZER_REGISTRY[n] for n in names_order},
    }}

    # === Section 1: Large-scale fertility & compression ===
    print("\n" + "=" * 90)
    print("  SECTION 1: FERTILITY & COMPRESSION (Large-Scale)")
    print("=" * 90)

    metrics = {}
    for label, data in [("TR", tr_data), ("EN", en_data), ("CS", cs_data)]:
        fert = {}
        comp = {}
        total_tok = {}
        print(f"\n  Computing on {len(data):,} {label} sentences...", flush=True)
        for name in names_order:
            tok, _ = all_tok[name]
            t0 = time.time()
            fert[name] = calc_fertility_bulk(tok, data)
            comp[name] = calc_compression_bulk(tok, data)
            total_tok[name] = calc_total_tokens(tok, data)
            dt = time.time() - t0
            print(f"    {SHORT_NAMES[name]:<12} fert={fert[name]:.2f}  "
                  f"comp={comp[name]:.2f}  "
                  f"tokens={total_tok[name]:>10,}  ({dt:.1f}s)")
        metrics[f"{label}_fertility"] = {n: round(fert[n], 3) for n in names_order}
        metrics[f"{label}_compression"] = {n: round(comp[n], 3) for n in names_order}
        metrics[f"{label}_total_tokens"] = {n: total_tok[n] for n in names_order}

    results["metrics"] = metrics

    # Summary table
    print("\n  --- Summary Table ---")
    print_table_header(names_order)
    for label in ["TR", "EN", "CS"]:
        print(fmt_table_row(
            f"{label} Fertility (tok/word)",
            metrics[f"{label}_fertility"], names_order))
        print(fmt_table_row(
            f"{label} Compression (chr/tok)",
            metrics[f"{label}_compression"], names_order))
        print(fmt_table_row(
            f"{label} Total Tokens",
            metrics[f"{label}_total_tokens"], names_order, fmt=","))

    # === Section 2: Vocab info ===
    print("\n" + "=" * 90)
    print("  SECTION 2: VOCABULARY COMPARISON")
    print("=" * 90)
    print_table_header(names_order)
    display_vocab_sizes = {n: get_vocab_size_for_display(n, all_tok) for n in names_order}
    print(fmt_table_row("Vocab Size", display_vocab_sizes, names_order, fmt=","))
    print(fmt_table_row("Algorithm",
          {n: TOKENIZER_REGISTRY[n]["algo"] for n in names_order}, names_order))

    # Vocab efficiency: tokens produced per vocab entry (lower = more efficient use of vocab)
    tr_eff = {}
    for n in names_order:
        tr_eff[n] = round(metrics["TR_total_tokens"][n] / display_vocab_sizes[n], 2)
    print(fmt_table_row("TR tok/vocab ratio", tr_eff, names_order))

    results["vocab_efficiency"] = tr_eff

    # === Section 3: Morphological splitting ===
    print("\n" + "=" * 90)
    print("  SECTION 3: TURKISH MORPHOLOGICAL SPLITTING")
    print("=" * 90)
    print("  Note: token lists below are raw tokenizer pieces; byte-level models may show UTF-8 fragments.")

    morph_results = []
    morph_totals = {n: 0 for n in names_order}

    for word, ideal in TR_MORPHOLOGY_WORDS:
        entry = {"word": word, "ideal": ideal}
        print(f"\n  {word}")
        print(f"    {'Ideal:':<14} {ideal}")
        for name in names_order:
            tok, _ = all_tok[name]
            tokens = tokenize_word(tok, word)
            count = len(tokens)
            morph_totals[name] += count
            token_str = " + ".join(tokens)
            print(f"    {SHORT_NAMES[name]:<14} {token_str:<55} [{count} tok]")
            entry[name] = {"tokens": tokens, "count": count}
        morph_results.append(entry)

    results["morphology"] = morph_results

    # Morph summary
    print(f"\n  {'--- Morphological Token Counts ---':^90}")
    w_col = 42
    hdr = f"  {'Word':<{w_col}}"
    for n in names_order:
        hdr += f" {SHORT_NAMES[n]:>10}"
    print(hdr)
    print("  " + "-" * (w_col + 11 * len(names_order)))
    for entry in morph_results:
        row = f"  {entry['word']:<{w_col}}"
        for n in names_order:
            row += f" {entry[n]['count']:>10}"
        print(row)
    row = f"  {'TOTAL':<{w_col}}"
    for n in names_order:
        row += f" {morph_totals[n]:>10}"
    print("  " + "-" * (w_col + 11 * len(names_order)))
    print(row)
    avg_row = f"  {'AVERAGE':<{w_col}}"
    for n in names_order:
        avg_row += f" {morph_totals[n]/len(morph_results):>10.1f}"
    print(avg_row)

    results["morph_totals"] = morph_totals

    # === Section 4: I/i normalization ===
    print("\n" + "=" * 90)
    print("  SECTION 4: TURKISH I/i NORMALIZATION")
    print("=" * 90)

    norm_scores = {n: 0 for n in names_order}
    norm_details = []

    hdr = f"  {'Input':<10} {'Expected':<10}"
    for n in names_order:
        hdr += f" {SHORT_NAMES[n]:>12}"
    print(hdr)
    print("  " + "-" * (20 + 13 * len(names_order)))

    for word, expected, rule in TURKISH_I_CASES:
        row = f"  {word:<10} {expected:<10}"
        entry = {"input": word, "expected": expected, "rule": rule}
        for name in names_order:
            _, raw = all_tok[name]
            result = check_normalization(raw, word)
            ok = result == expected
            if ok:
                norm_scores[name] += 1
            mark = "OK" if ok else result
            row += f" {mark:>12}"
            entry[name] = {"result": result, "correct": ok}
        print(row)
        norm_details.append(entry)

    total_cases = len(TURKISH_I_CASES)
    score_row = f"  {'SCORE':<10} {'':10}"
    for n in names_order:
        score_row += f" {f'{norm_scores[n]}/{total_cases}':>12}"
    print("  " + "-" * (20 + 13 * len(names_order)))
    print(score_row)

    results["normalization"] = {
        "scores": {n: f"{norm_scores[n]}/{total_cases}" for n in names_order},
        "details": norm_details,
    }

    # === Section 5: Code-switching ===
    print("\n" + "=" * 90)
    print("  SECTION 5: CODE-SWITCHING TOKENIZATION EXAMPLES")
    print("=" * 90)
    print("  Note: raw token pieces are shown without decode cleanup; counts remain exact.")

    cs_examples = []
    for sent in CS_SENTENCES[:5]:  # show 5 examples in detail
        entry = {"sentence": sent}
        print(f"\n  \"{sent}\"")
        for name in names_order:
            tok, _ = all_tok[name]
            tokens = tok.tokenize(sent)
            count = len(tokens)
            display = [clean_display_token(t) or t for t in tokens]
            print(f"    {SHORT_NAMES[name]:<14} [{count:>2} tok] {display}")
            entry[name] = {"tokens": display, "count": count}
        cs_examples.append(entry)

    results["code_switching_examples"] = cs_examples

    # === Section 6: Final summary ===
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print_table_header(names_order)
    print(fmt_table_row("Vocab Size", display_vocab_sizes, names_order, fmt=","))
    print(fmt_table_row("Algorithm",
          {n: TOKENIZER_REGISTRY[n]["algo"] for n in names_order}, names_order))
    print(fmt_table_row("TR Fertility",
          metrics["TR_fertility"], names_order))
    print(fmt_table_row("EN Fertility",
          metrics["EN_fertility"], names_order))
    print(fmt_table_row("TR Compression",
          metrics["TR_compression"], names_order))
    print(fmt_table_row("EN Compression",
          metrics["EN_compression"], names_order))
    print(fmt_table_row("TR Total Tokens (5K sent)",
          metrics["TR_total_tokens"], names_order, fmt=","))
    print(fmt_table_row("EN Total Tokens (5K sent)",
          metrics["EN_total_tokens"], names_order, fmt=","))
    print(fmt_table_row("Morph Tokens (10 words)",
          morph_totals, names_order, fmt=","))
    print(fmt_table_row("I/i Normalization",
          {n: f"{norm_scores[n]}/{total_cases}" for n in names_order}, names_order))

    results["summary"] = {
        "vocab_sizes": display_vocab_sizes,
        "tr_fertility": metrics["TR_fertility"],
        "en_fertility": metrics["EN_fertility"],
        "tr_compression": metrics["TR_compression"],
        "en_compression": metrics["EN_compression"],
        "tr_total_tokens": metrics["TR_total_tokens"],
        "en_total_tokens": metrics["EN_total_tokens"],
        "morph_totals": morph_totals,
        "normalization_scores": {n: norm_scores[n] for n in names_order},
        "normalization_total": total_cases,
    }

    # Save
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to {RESULTS_PATH}")
    print("=" * 90)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="Multrenizer Large-Scale Benchmark")
    parser.add_argument("--tr-lines", type=int, default=5000,
                        help="Number of TR sentences to benchmark (default: 5000)")
    parser.add_argument("--en-lines", type=int, default=5000,
                        help="Number of EN sentences to benchmark (default: 5000)")
    parser.add_argument("--cs-lines", type=int, default=500,
                        help="Number of CS sentences to benchmark (default: 500)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Data directory from prepare_data.py (default: data/)")
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    run_benchmark(
        tr_lines=args.tr_lines,
        en_lines=args.en_lines,
        cs_lines=args.cs_lines,
    )


if __name__ == "__main__":
    main()
