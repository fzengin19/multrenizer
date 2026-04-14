#!/usr/bin/env python3
"""
Multrenizer - Bilingual EN-TR Unigram Tokenizer Trainer
========================================================
Trains a custom Unigram tokenizer optimized for English and Turkish
with locale-aware Turkish normalization, apostrophe-aware pre-tokenization,
and code-switching support.

Target vocab size: 26,000 (836 fixed + up to 25,164 learned subword)
Current artifact size may land slightly below target depending on trainer convergence.
Algorithm: Unigram Language Model
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
from huggingface_hub import HfApi, login


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

VOCAB_SIZE = 26_000
UNK_TOKEN = "<unk>"

# --- Special Tokens ---
# 32 named (ID 0-31) + 512 reserved (ID 32-543) + 292 utility (ID 544+)
# = 836 total fixed/special tokens
# Remaining 25,164 slots are available for learned subword tokens.
#
# Vocab budget:
#   26,000 total
#    - 32  named special   (chat, reasoning, tool-use, etc.)
#    - 512 reserved        (future use without retraining)
#    - 292 utility         (emoji, punctuation, symbols)
#    = 25,164 subword      (learned from corpus)

_NAMED_SPECIAL_TOKENS = [
    # Core (0-3)
    "<unk>",                # 0  - Unknown / OOV fallback
    "<s>",                  # 1  - Beginning of sequence (BOS)
    "</s>",                 # 2  - End of sequence (EOS)
    "<pad>",                # 3  - Padding for batch training

    # Chat / Instruction (4-8)
    "<|system|>",           # 4  - System prompt start
    "<|user|>",             # 5  - User turn start
    "<|assistant|>",        # 6  - Assistant turn start
    "<|end|>",              # 7  - End of turn / message boundary
    "<|sep|>",              # 8  - Generic separator (sentence pairs, etc.)

    # Reasoning / Chain-of-Thought (9-12)
    "<think>",              # 9  - Start internal reasoning (DeepSeek-R1 / Qwen style)
    "</think>",             # 10 - End internal reasoning
    "<|step|>",             # 11 - Reasoning step boundary
    "<|reflection|>",       # 12 - Self-verification / backtrack marker

    # Tool Use / Function Calling (13-16)
    "<tool_call>",          # 13 - Tool invocation start
    "</tool_call>",         # 14 - Tool invocation end
    "<tool_response>",      # 15 - Tool result start
    "</tool_response>",     # 16 - Tool result end

    # Code (17-20)
    "<|code|>",             # 17 - Code-switching / programming code marker
    "<|fim_prefix|>",       # 18 - Fill-in-middle: prefix (code completion)
    "<|fim_middle|>",       # 19 - Fill-in-middle: middle (cursor position)
    "<|fim_suffix|>",       # 20 - Fill-in-middle: suffix

    # Bilingual / Language Tags (21-22)
    "<|tr|>",               # 21 - Turkish content marker
    "<|en|>",               # 22 - English content marker

    # RAG / Retrieval (23-24)
    "<|context|>",          # 23 - Retrieved context start
    "<|/context|>",         # 24 - Retrieved context end

    # Multi-modal Placeholders (25-28)
    "<|image|>",            # 25 - Image placeholder
    "<|audio|>",            # 26 - Audio placeholder
    "<|video|>",            # 27 - Video placeholder
    "<|file|>",             # 28 - File/document attachment

    # Structured Output (29-31)
    "<|json|>",             # 29 - JSON output marker
    "<|table|>",            # 30 - Table/structured data marker
    "<|cite|>",             # 31 - Citation / source reference marker
]

_RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(512)]  # ID 32-543

# --- Utility Tokens (ID 544+) ---
# Hand-picked tokens for emoji, punctuation, symbols, and common characters
# that must always be preserved (never dropped by pre-tokenizer).
# These get dedicated IDs so the model can learn their semantics.

_PUNCTUATION_TOKENS = [
    ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}",
    "/", "\\", '"', "'", "...", "\u2013", "\u2014", "\u2026", "_", "*",
    "+", "=", "<", ">", "|", "~", "^", "`",
]

_CURRENCY_SYMBOLS = [
    "\u20ba", "$", "\u20ac", "\u00a3", "\u00a5", "\u20b9",  # ₺ $ € £ ¥ ₹
    "%", "\u2030", "\u00b0", "\u00a7", "\u00b6", "\u2116",  # % ‰ ° § ¶ №
    "@", "#", "&",
]

_MATH_SYMBOLS = [
    "\u00b1", "\u00d7", "\u00f7", "\u2260", "\u2264", "\u2265",  # ± × ÷ ≠ ≤ ≥
    "\u2248", "\u221e", "\u221a", "\u2211", "\u222b", "\u2202",  # ≈ ∞ √ ∑ ∫ ∂
    "\u0394", "\u03c0", "\u03b1", "\u03b2", "\u03b3", "\u03b4",  # Δ π α β γ δ
    "\u03b5", "\u03b8", "\u03bb", "\u03bc", "\u03c3", "\u03c6", "\u03c9",  # ε θ λ μ σ φ ω
]

_ARROW_SYMBOLS = [
    "\u2192", "\u2190", "\u2191", "\u2193", "\u2194", "\u21d2",  # → ← ↑ ↓ ↔ ⇒
    "\u2022", "\u00b7", "\u2605", "\u2606",                      # • · ★ ☆
    "\u2713", "\u2717", "\u00a9", "\u00ae", "\u2122",            # ✓ ✗ © ® ™
]

_TYPOGRAPHY_TOKENS = [
    "\u00ab", "\u00bb", "\u201c", "\u201d", "\u2018", "\u2019",  # « » " " ' '
    "\u2039", "\u203a", "\u201e", "\u201a",                      # ‹ › „ ‚
]

_EMOJI_FACES = [
    "\U0001f600", "\U0001f603", "\U0001f604", "\U0001f601", "\U0001f606",
    "\U0001f605", "\U0001f923", "\U0001f602", "\U0001f642", "\U0001f60a",
    "\U0001f607", "\U0001f970", "\U0001f60d", "\U0001f929", "\U0001f618",
    "\U0001f617", "\U0001f61a", "\U0001f619", "\U0001f972", "\U0001f60b",
    "\U0001f61b", "\U0001f61c", "\U0001f92a", "\U0001f61d", "\U0001f911",
    "\U0001f917", "\U0001f92d", "\U0001f92b", "\U0001f914", "\U0001f910",
    "\U0001f928", "\U0001f610", "\U0001f611", "\U0001f636", "\U0001f60f",
    "\U0001f612", "\U0001f644", "\U0001f62c", "\U0001f62e", "\U0001f632",
    "\U0001f633", "\U0001f97a", "\U0001f626", "\U0001f627", "\U0001f628",
    "\U0001f630", "\U0001f625", "\U0001f622", "\U0001f62d", "\U0001f631",
    "\U0001f616", "\U0001f623", "\U0001f61e", "\U0001f613", "\U0001f629",
    "\U0001f62b", "\U0001f971", "\U0001f624", "\U0001f621", "\U0001f620",
    "\U0001f92c", "\U0001f608", "\U0001f47f", "\U0001f480", "\U0001f4a9",
    "\U0001f921", "\U0001f47b", "\U0001f47d", "\U0001f47e", "\U0001f916",
]

_EMOJI_HANDS = [
    "\U0001f44b", "\U0001f91a", "\u270b", "\U0001f596",
    "\U0001f44c", "\U0001f90c", "\U0001f90f", "\u270c\ufe0f",
    "\U0001f91e", "\U0001f91f", "\U0001f918", "\U0001f919",
    "\U0001f448", "\U0001f449", "\U0001f446", "\U0001f595",
    "\U0001f447", "\u261d\ufe0f", "\U0001f44d", "\U0001f44e",
    "\u270a", "\U0001f44a", "\U0001f91b", "\U0001f91c",
    "\U0001f44f", "\U0001f64c", "\U0001f64f", "\U0001f4aa",
]

_EMOJI_HEARTS = [
    "\u2764\ufe0f", "\U0001f9e1", "\U0001f49b", "\U0001f49a",
    "\U0001f499", "\U0001f49c", "\U0001f5a4", "\U0001f90d",
    "\U0001f90e", "\U0001f494", "\u2763\ufe0f", "\U0001f495",
    "\U0001f49e", "\U0001f493", "\U0001f497", "\U0001f496",
    "\U0001f498", "\U0001f49d",
]

_EMOJI_SYMBOLS = [
    "\U0001f525", "\u2728", "\U0001f31f", "\U0001f4ab", "\u2b50",
    "\U0001f308", "\u2600\ufe0f", "\U0001f319", "\u26a1", "\u2744\ufe0f",
    "\U0001f30a", "\U0001f338", "\U0001f33b", "\U0001f340",
    "\U0001f389", "\U0001f38a", "\U0001f388", "\U0001f381",
    "\u2705", "\u274c", "\u26a0\ufe0f", "\U0001f6ab",
    "\u2757", "\u2753", "\u203c\ufe0f", "\u2049\ufe0f",
    "\U0001f4af", "\U0001f534", "\U0001f7e0", "\U0001f7e1",
    "\U0001f7e2", "\U0001f535", "\U0001f7e3", "\U0001f7e4",
    "\u26ab", "\u26aa",
]

_EMOJI_OBJECTS = [
    "\U0001f680", "\U0001f4bb", "\U0001f4f1", "\U0001f4a1", "\U0001f511",
    "\U0001f512", "\U0001f513", "\U0001f4cc", "\U0001f4cd",
    "\U0001f4ca", "\U0001f4c8", "\U0001f4c9", "\U0001f3af",
    "\U0001f3c6", "\U0001f947", "\U0001f948", "\U0001f949",
    "\U0001f4f8", "\U0001f3b5", "\U0001f3b6", "\U0001f3a7",
    "\U0001f4da", "\u270f\ufe0f", "\U0001f4dd", "\U0001f4cb",
    "\U0001f4ce", "\U0001f517", "\U0001f4b0", "\U0001f4b5",
    "\u2615", "\U0001f355", "\U0001f354", "\U0001f382",
    "\U0001f370", "\U0001f369", "\U0001f36b",
]

_EMOJI_FLAGS = [
    "\U0001f1f9\U0001f1f7",  # 🇹🇷
    "\U0001f1fa\U0001f1f8",  # 🇺🇸
    "\U0001f1ec\U0001f1e7",  # 🇬🇧
    "\U0001f1e9\U0001f1ea",  # 🇩🇪
    "\U0001f1eb\U0001f1f7",  # 🇫🇷
    "\U0001f1ea\U0001f1f8",  # 🇪🇸
    "\U0001f1ee\U0001f1f9",  # 🇮🇹
    "\U0001f1ef\U0001f1f5",  # 🇯🇵
]

_UTILITY_TOKENS = (
    _PUNCTUATION_TOKENS
    + _CURRENCY_SYMBOLS
    + _MATH_SYMBOLS
    + _ARROW_SYMBOLS
    + _TYPOGRAPHY_TOKENS
    + _EMOJI_FACES
    + _EMOJI_HANDS
    + _EMOJI_HEARTS
    + _EMOJI_SYMBOLS
    + _EMOJI_OBJECTS
    + _EMOJI_FLAGS
)

SPECIAL_TOKENS = _NAMED_SPECIAL_TOKENS + _RESERVED_TOKENS + _UTILITY_TOKENS

# Corpus mixing hints used by the demo iterator and data-prep metadata.
# The released artifact is typically trained with the file-based 6/3/1
# interleave defined later in this script (~60% TR / ~30% EN / ~10% CS).
TR_RATIO = 0.65
EN_RATIO = 0.35
CODE_SWITCH_RATIO = 0.10  # portion of total that is synthetic code-switching


# ---------------------------------------------------------------------------
# 2. Turkish Locale-Aware Normalizer
# ---------------------------------------------------------------------------

def build_turkish_normalizer():
    """
    Build a normalizer that correctly handles Turkish dotted/dotless I
    BEFORE any standard Unicode normalization or lowercasing.

    Turkish rules:
      - Upper 'I'  -> lower 'ı'  (dotless)
      - Upper 'İ'  -> lower 'i'  (dotted)

    Standard Python/Unicode lowering gets this wrong, producing
    'COMBINING DOT ABOVE' (U+0307) artifacts. We fix it with explicit
    Replace steps that run FIRST in the Sequence.
    """
    return normalizers.Sequence([
        # Canonicalize common apostrophe/quote variants before tokenization.
        normalizers.Replace("\u2018", "'"),
        normalizers.Replace("\u2019", "'"),
        normalizers.Replace("\u02bc", "'"),
        normalizers.Replace("\uff07", "'"),
        # --- Turkish-specific uppercase -> lowercase mappings ---
        # Must run BEFORE any generic lowercasing / NFC / NFKC.
        normalizers.Replace("İ", "i"),   # U+0130 -> i
        normalizers.Replace("I", "ı"),   # U+0049 -> ı (U+0131)
        # Handle the decomposed form: i + U+0307 -> i
        normalizers.Replace("i\u0307", "i"),
        # --- Standard normalization ---
        normalizers.NFKC(),
        normalizers.Lowercase(),
        normalizers.Strip(),
    ])


# ---------------------------------------------------------------------------
# 3. Pre-Tokenizer with Apostrophe Awareness
# ---------------------------------------------------------------------------

def build_pre_tokenizer():
    """
    Pre-tokenizer that:
    - Splits on whitespace first
    - Preserves apostrophes as standalone tokens
      (merge'lemek -> merge + ' + lemek)
    - Isolates remaining non-letter/non-digit characters as their own tokens
      (punctuation, emoji, symbols are KEPT, not dropped)
    """
    return pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        # Preserve apostrophes as standalone tokens so contractions,
        # possessives, and Turkish proper-noun suffixes are not destroyed.
        pre_tokenizers.Split(
            pattern=Regex(r"'"),
            behavior="isolated",
            invert=False,
        ),
        # Keep other non-letter/non-digit chars as isolated tokens
        # (punctuation, emoji, symbols get their own token slots)
        pre_tokenizers.Split(
            pattern=Regex(r"[^\p{L}\p{N}]"),
            behavior="isolated",
            invert=False,
        ),
    ])


# ---------------------------------------------------------------------------
# 4. Sample Corpus Data (Built-in Demo)
# ---------------------------------------------------------------------------

# Representative Turkish texts covering various registers
SAMPLE_TR_TEXTS = [
    # News / formal register
    "Cumhurbaşkanı, yeni ekonomi paketini bugün açıkladı. Paket kapsamında vergi indirimleri ve istihdam teşvikleri yer alıyor.",
    "Türkiye Büyük Millet Meclisi, yeni yasama dönemine başladı. Milletvekilleri komisyon çalışmalarını hızlandırdı.",
    "Merkez Bankası faiz kararını açıkladı. Politika faizi yüzde yirmi beş olarak sabit tutuldu.",
    "İstanbul'da düzenlenen uluslararası konferansta yapay zeka teknolojileri tartışıldı.",
    "Deprem bölgesinde yeniden yapılanma çalışmaları hız kesmeden devam ediyor.",
    # Conversational / informal
    "Yarın akşam buluşalım mı? Kadıköy'de güzel bir kafe açılmış, oraya gidelim.",
    "Dün gece maçı izledin mi? İnanılmaz bir gol attı, herkes şaşırdı.",
    "Anneannem en güzel mantıyı yapar, kimse onun tarifine yetişemez.",
    "Bu kitabı okuduktan sonra hayata bakış açım tamamen değişti.",
    "Çocukluğumda her yaz köye giderdik, dedemin bahçesinde oynardık.",
    # Academic / technical
    "Makine öğrenmesi algoritmalarının doğal dil işleme üzerindeki etkisi son yıllarda katlanarak artmıştır.",
    "Morfolojik analiz, sondan eklemeli dillerde tokenizasyon kalitesini doğrudan etkileyen kritik bir adımdır.",
    "Ünlü uyumu ve ünsüz benzeşmesi Türkçenin temel morfofonolojik kurallarıdır.",
    "Transformer mimarisinin dikkat mekanizması, uzun mesafeli bağımlılıkları yakalamada etkilidir.",
    "Derin öğrenme modellerinin eğitiminde veri kalitesi, model boyutundan daha belirleyici olabilir.",
    # Agglutinative morphology showcase
    "Afyonkarahisarlılaştıramadıklarımızdan mısınız? Bu kelimedeki eklerin her biri ayrı bir anlam taşır.",
    "Evlerimizdekilerin hepsini göremedik, onları ziyaret etmeliydik.",
    "Çalışmalarımızın sonuçlarını değerlendirememişlerdi, tekrar incelenmesi gerekiyordu.",
    "Okuduklarımdan anladığıma göre bu konuda uzlaşma sağlanamamış.",
    "Güzelleştirilmiş bahçelerdeki çiçeklerin kokusunu duyabiliyordunuz.",
    # Literature / cultural
    "İstanbul'un tarihi sokaklarında yürürken geçmişin izlerini her adımda hissedersiniz.",
    "Anadolu'nun bereketli toprakları binlerce yıldır medeniyetlere ev sahipliği yapmıştır.",
    "Türk mutfağının zenginliği, coğrafi çeşitlilikten ve kültürel etkileşimlerden kaynaklanmaktadır.",
    "Nasreddin Hoca fıkraları, yüzyıllardır toplumsal eleştiriyi mizahla harmanlayan eşsiz örneklerdir.",
    "Orhan Pamuk'un romanlarında İstanbul'un çok katmanlı tarihi ustalıkla işlenir.",
]

# Representative English texts
SAMPLE_EN_TEXTS = [
    # News / formal
    "The Federal Reserve announced its decision to maintain interest rates at their current level amid ongoing economic uncertainty.",
    "Climate scientists reported that global temperatures reached a new record high in the past decade.",
    "The United Nations General Assembly convened to discuss international cooperation on sustainable development goals.",
    "Artificial intelligence continues to transform industries from healthcare to financial services.",
    "The European Union proposed new regulations to govern the ethical use of machine learning systems.",
    # Technical / academic
    "Natural language processing has evolved significantly with the advent of transformer architectures and attention mechanisms.",
    "Subword tokenization algorithms such as BPE and Unigram enable models to handle open vocabularies efficiently.",
    "The embedding layer in small language models can constitute up to thirty percent of total model parameters.",
    "Cross-lingual transfer learning leverages shared representations across languages to improve low-resource performance.",
    "Morphological alignment metrics measure how well tokenizer boundaries correspond to linguistic morpheme boundaries.",
    # Conversational
    "Have you tried the new restaurant downtown? The food is amazing and the prices are reasonable.",
    "I just finished reading that book you recommended. The ending completely caught me off guard.",
    "Let's meet up this weekend. There's a great exhibition at the museum that closes on Sunday.",
    "The kids had so much fun at the park yesterday. They didn't want to come home.",
    "My grandmother's apple pie recipe has been passed down through four generations of our family.",
]

# Synthetic code-switching texts (TR-EN mix)
SAMPLE_CODE_SWITCH_TEXTS = [
    # Developer / tech jargon (plaza dili)
    "Deploy sürecinde unexpected bir error aldık, rollback yapmamız gerekiyor.",
    "Bu feature'ı implement ederken edge case'leri handle etmeyi unutmayalım.",
    "Sprint planning'de discuss ettiğimiz gibi, backend refactoring'i önceliklendirmeliyiz.",
    "Production'a push etmeden önce staging'de test edelim, last time hotfix çıkarmak zorunda kaldık.",
    "Code review'da feedback aldım, design pattern'ı değiştirmemiz lazım.",
    "Bu API endpoint'i optimize etmemiz gerekiyor, response time çok yüksek.",
    "Database migration'ı run ettikten sonra cache'i invalidate etmeyi unutma.",
    "Kubernetes cluster'ında resource limit'leri adjust etmemiz lazım.",
    "CI/CD pipeline'ında build fail oluyor, dependency conflict var sanırım.",
    "Microservice architecture'a geçiş planını draft ettim, review eder misin?",
    # Business / corporate
    "Meeting'i reschedule edelim, stakeholder'lar available değil bugün.",
    "Q3 target'larımızı exceed ettik ama Q4 forecast biraz conservative.",
    "Deadline'a kadar deliverable'ları finalize etmemiz gerekiyor, aksi halde milestone'u kaçırırız.",
    "Budget approval'ı aldık, procurement process'i başlatabiliriz.",
    "Performance review'da feedback olarak leadership skill'lerimi geliştirmem söylendi.",
    # Social media / informal
    "Bu outfit çok aesthetic olmuş, where did you get it?",
    "Vibe'ı çok iyi olan bir cafe keşfettim, literally her köşesi Instagram'lık.",
    "Bence bu trend overrated, insanlar sadece hype'a kapılıyor.",
    "Content creation işi sanıldığı kadar easy değil, çok effort gerektiriyor.",
    "Playlist'ime yeni şarkılar ekledim, shuffle'da çok güzel çalıyor.",
    # Scrambled / devrik syntax (for robustness)
    "Error unexpected bir deploy sürecinde aldık biz.",
    "Etmeyi handle edge case'leri unutmayalım implement ederken.",
    "Önceliklendirmeliyiz backend refactoring'i, planning'de discuss etmiştik.",
    "Push etmeden production'a, staging'de mutlaka test edilmeli.",
    "Gerekiyor optimize etmemiz endpoint'i, time response kabul edilemez.",
]


def generate_corpus_iterator(
    tr_texts: list[str],
    en_texts: list[str],
    cs_texts: list[str],
    *,
    repeat_factor: int = 50,
    chunk_size: int = 100,
):
    """
    Memory-friendly generator that yields text chunks from a mixed corpus.

    This helper is meant for demo/smoke-test training when texts are already
    preselected in the desired balance. Exact output ratios depend on the
    caller-provided list sizes.

    Args:
        tr_texts: List of Turkish text samples.
        en_texts: List of English text samples.
        cs_texts: List of code-switching text samples.
        repeat_factor: How many times to repeat the base texts (to simulate
                       a larger corpus for the demo).
        chunk_size: Number of lines per yielded batch.
    """
    # Build the full pool by repeating the provided lists as-is.
    pool: list[str] = []

    for _ in range(repeat_factor):
        pool.extend(tr_texts)
        pool.extend(en_texts)
        pool.extend(cs_texts)

    random.shuffle(pool)

    # Yield in chunks (memory-friendly)
    chunk: list[str] = []
    for line in pool:
        chunk.append(line)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def generate_corpus_from_files(
    tr_paths: list[str],
    en_paths: list[str],
    cs_paths: list[str],
    chunk_size: int = 1000,
):
    """
    Production-grade generator that reads corpus files from disk in chunks.

    Each file is read line by line. Lines from different sources are interleaved
    with a fixed 6/3/1 schedule, yielding an effective mix of roughly
    60% Turkish, 30% English, and 10% code-switching.

    Args:
        tr_paths: Paths to Turkish corpus files.
        en_paths: Paths to English corpus files.
        cs_paths: Paths to code-switching corpus files.
        chunk_size: Number of lines per yielded batch.
    """
    def _line_reader(paths: list[str]):
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        yield stripped

    tr_iter = _line_reader(tr_paths)
    en_iter = _line_reader(en_paths)
    cs_iter = _line_reader(cs_paths)

    chunk: list[str] = []

    # Interleave using a weighted round-robin: 6 TR / 3 EN / 1 CS.
    weights = {"tr": 6, "en": 3, "cs": 1}
    sources = {
        "tr": tr_iter,
        "en": en_iter,
        "cs": cs_iter,
    }

    exhausted: set[str] = set()

    while len(exhausted) < len(sources):
        for lang, count in weights.items():
            if lang in exhausted:
                continue
            for _ in range(count):
                try:
                    line = next(sources[lang])
                    chunk.append(line)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                except StopIteration:
                    exhausted.add(lang)
                    break

    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# 5. Tokenizer Builder
# ---------------------------------------------------------------------------

def build_tokenizer() -> Tokenizer:
    """
    Assemble the full tokenizer pipeline:
      1. Unigram model (empty, to be trained)
      2. Turkish locale-aware normalizer
      3. Apostrophe + punctuation pre-tokenizer
      4. Post-processor with <s> / </s> template
    """
    tokenizer = Tokenizer(models.Unigram())

    # Normalizer: Turkish I/i fix -> NFKC -> lowercase -> strip
    tokenizer.normalizer = build_turkish_normalizer()

    # Pre-tokenizer: whitespace + apostrophe + punctuation split
    tokenizer.pre_tokenizer = build_pre_tokenizer()

    # Post-processor: wrap sequences with <s> ... </s>
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", 1),
            ("</s>", 2),
        ],
    )

    return tokenizer


def train_tokenizer(
    tokenizer: Tokenizer,
    corpus_iterator,
    vocab_size: int = VOCAB_SIZE,
) -> Tokenizer:
    """
    Train the Unigram model on the provided corpus iterator.

    Args:
        tokenizer: Pre-configured Tokenizer with normalizer/pre-tokenizer.
        corpus_iterator: Generator yielding batches of text (list[str]).
        vocab_size: Target vocabulary size (default: 26,000).

    Returns:
        The trained Tokenizer.
    """
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token=UNK_TOKEN,
        shrinking_factor=0.75,
        max_piece_length=24,
        n_sub_iterations=2,
    )

    # Train from iterator (memory-friendly: processes chunk by chunk)
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    return tokenizer


# ---------------------------------------------------------------------------
# 6. Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_tokenizer(tokenizer: Tokenizer):
    """Print basic evaluation metrics for the trained tokenizer."""

    print("\n" + "=" * 70)
    print("TOKENIZER EVALUATION")
    print("=" * 70)

    vocab_size = tokenizer.get_vocab_size()
    print(f"\nVocab size: {vocab_size:,}")

    # Test Turkish morphology
    tr_test_cases = [
        "evlerimizdekilerin",
        "güzelleştirilmiş",
        "çalışmalarımızın",
        "Afyonkarahisarlılaştıramadıklarımızdan",
        "okuduklarımdan",
        "İstanbul'da",
    ]

    print("\n--- Turkish Morphology Test ---")
    total_tr_tokens = 0
    for word in tr_test_cases:
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        total_tr_tokens += len(tokens)
        print(f"  {word:<45} -> {tokens}")

    avg_fertility_tr = total_tr_tokens / len(tr_test_cases)
    print(f"\n  Avg Turkish fertility (tokens/word): {avg_fertility_tr:.2f}")

    # Test English
    en_test_cases = [
        "understanding",
        "transformers",
        "tokenization",
        "computational",
        "morphological",
        "sustainability",
    ]

    print("\n--- English Morphology Test ---")
    total_en_tokens = 0
    for word in en_test_cases:
        encoded = tokenizer.encode(word)
        tokens = encoded.tokens
        total_en_tokens += len(tokens)
        print(f"  {word:<45} -> {tokens}")

    avg_fertility_en = total_en_tokens / len(en_test_cases)
    print(f"\n  Avg English fertility (tokens/word): {avg_fertility_en:.2f}")

    # Test code-switching
    cs_test_cases = [
        "Deploy sürecinde unexpected bir error aldık",
        "Bu feature'ı implement ederken dikkatli olalım",
        "merge'lemek istediğim branch conflict veriyor",
    ]

    print("\n--- Code-Switching Test ---")
    for sent in cs_test_cases:
        encoded = tokenizer.encode(sent)
        tokens = encoded.tokens
        print(f"  {sent}")
        print(f"    -> {tokens}")
        print()

    # Test Turkish I/i normalization
    print("--- Turkish I/i Normalization Test ---")
    i_test_cases = [
        ("İstanbul", "istanbul expected"),
        ("IŞIK", "ışık expected"),
        ("İŞ", "iş expected"),
        ("SIR", "sır expected (not sir)"),
    ]
    for word, expected in i_test_cases:
        normalized = tokenizer.normalizer.normalize_str(word)
        encoded = tokenizer.encode(word)
        print(f"  {word:<20} -> normalized: '{normalized}' | tokens: {encoded.tokens}")

    # Compression ratio on a sample paragraph
    print("\n--- Compression Ratio ---")
    sample_tr = (
        "Türkiye'nin en büyük şehri olan İstanbul, tarihi ve kültürel "
        "zenginlikleriyle dünyaca ünlüdür. Boğaziçi Köprüsü Avrupa ile "
        "Asya'yı birbirine bağlar."
    )
    sample_en = (
        "Istanbul is the largest city in Turkey, renowned worldwide for "
        "its historical and cultural richness. The Bosphorus Bridge connects "
        "Europe and Asia."
    )

    for label, text in [("TR", sample_tr), ("EN", sample_en)]:
        encoded = tokenizer.encode(text)
        n_chars = len(text)
        n_tokens = len(encoded.tokens)
        ratio = n_chars / n_tokens if n_tokens > 0 else 0
        print(f"  [{label}] chars={n_chars}, tokens={n_tokens}, "
              f"compression={ratio:.2f} chars/token")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 7. Save & Push
# ---------------------------------------------------------------------------

def save_and_push(
    tokenizer: Tokenizer,
    output_dir: str,
    repo_id: str | None = None,
    hf_token: str | None = None,
):
    """
    Save the tokenizer locally and optionally push to Hugging Face Hub.

    Saves:
      - tokenizer.json (the full tokenizer state)
      - tokenizer_config.json (HF-compatible config)
      - special_tokens_map.json

    Args:
        tokenizer: Trained Tokenizer instance.
        output_dir: Local directory to save files.
        repo_id: HF Hub repo id (e.g. "username/multrenizer"). None to skip push.
        hf_token: HF API token. None to use cached login.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the raw tokenizer.json
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Build HF-compatible config files
    # additional_special_tokens = everything except the 4 core roles
    core = {"<unk>", "<s>", "</s>", "<pad>"}
    additional = [t for t in SPECIAL_TOKENS if t not in core]

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "unigram",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": additional,
        "clean_up_tokenization_spaces": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% if system_message != '' %}{{ '<|system|>\\n' + system_message + '<|end|>\\n' }}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
    }

    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"Saved tokenizer_config.json to {config_path}")

    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": additional,
    }

    special_path = os.path.join(output_dir, "special_tokens_map.json")
    with open(special_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    print(f"Saved special_tokens_map.json to {special_path}")

    # Push to Hub
    if repo_id:
        print(f"\nPushing to Hugging Face Hub: {repo_id}")
        if hf_token:
            login(token=hf_token)

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Multrenizer bilingual EN-TR Unigram tokenizer (~26K target vocab)",
        )
        print(f"Successfully pushed to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# 8. Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multrenizer: Train a bilingual EN-TR Unigram tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with built-in sample data (demo mode)
  python train_tokenizer.py --demo

  # Train with data prepared by prepare_data.py (auto-detects data/ dir)
  python prepare_data.py --size medium
  python train_tokenizer.py --data-dir data/

  # Train with custom corpus files
  python train_tokenizer.py \\
    --tr-corpus data/tr_corpus.txt \\
    --en-corpus data/en_corpus.txt \\
    --cs-corpus data/cs_corpus.txt

  # Train and push to Hugging Face Hub
  python train_tokenizer.py --data-dir data/ \\
    --repo-id username/multrenizer \\
    --hf-token hf_xxxxx
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use built-in sample texts for training (demonstration mode)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory produced by prepare_data.py (auto-detects tr/en/cs corpus files)",
    )
    parser.add_argument(
        "--tr-corpus",
        nargs="+",
        default=[],
        help="Path(s) to Turkish corpus text file(s), one sentence per line",
    )
    parser.add_argument(
        "--en-corpus",
        nargs="+",
        default=[],
        help="Path(s) to English corpus text file(s), one sentence per line",
    )
    parser.add_argument(
        "--cs-corpus",
        nargs="+",
        default=[],
        help="Path(s) to code-switching corpus text file(s), one sentence per line",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Vocabulary size (default: {VOCAB_SIZE:,})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./multrenizer-tokenizer",
        help="Directory to save the trained tokenizer (default: ./multrenizer-tokenizer)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face Hub repo ID for push_to_hub (e.g. username/multrenizer)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--repeat-factor",
        type=int,
        default=100,
        help="How many times to repeat demo texts to simulate larger corpus (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    args = parser.parse_args()

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Set random seed
    random.seed(args.seed)

    # Resolve --data-dir into individual corpus paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            files = manifest.get("files", {})
            stats = manifest.get("corpus_stats", {})
            print(f"  Loaded manifest from {manifest_path}")
            print(f"  Corpus: TR={stats.get('tr_lines',0):,} "
                  f"EN={stats.get('en_lines',0):,} "
                  f"CS={stats.get('cs_lines',0):,}")
        if not args.tr_corpus:
            tr_file = data_dir / "tr_corpus.txt"
            if tr_file.exists():
                args.tr_corpus = [str(tr_file)]
        if not args.en_corpus:
            en_file = data_dir / "en_corpus.txt"
            if en_file.exists():
                args.en_corpus = [str(en_file)]
        if not args.cs_corpus:
            cs_file = data_dir / "cs_corpus.txt"
            if cs_file.exists():
                args.cs_corpus = [str(cs_file)]

    # Validate inputs
    use_demo = args.demo or (not args.tr_corpus and not args.en_corpus)
    if not use_demo and not args.tr_corpus:
        parser.error("Provide --data-dir, --tr-corpus/--en-corpus, or use --demo mode")

    print("=" * 70)
    print("  MULTRENIZER - Bilingual EN-TR Unigram Tokenizer Trainer")
    print("=" * 70)
    print(f"  Vocab size:   {args.vocab_size:,}")
    print(f"  Algorithm:    Unigram Language Model")
    print(f"  Mode:         {'Demo (built-in samples)' if use_demo else 'Custom corpus'}")
    print(f"  Output:       {args.output_dir}")
    if args.repo_id:
        print(f"  Hub repo:     {args.repo_id}")
    print("=" * 70)

    # Step 1: Build tokenizer pipeline
    print("\n[1/4] Building tokenizer pipeline...")
    tokenizer = build_tokenizer()
    print("  - Normalizer:     Turkish I/i fix -> NFKC -> Lowercase -> Strip")
    print("  - Pre-tokenizer:  Whitespace + Apostrophe + Punctuation split")
    print("  - Post-processor: <s> ... </s> template")
    print("  - Model:          Unigram (untrained)")

    # Step 2: Prepare corpus iterator
    print("\n[2/4] Preparing corpus...")
    if use_demo:
        print(f"  Using built-in samples (repeat_factor={args.repeat_factor})")
        print(f"  Turkish samples:       {len(SAMPLE_TR_TEXTS)}")
        print(f"  English samples:       {len(SAMPLE_EN_TEXTS)}")
        print(f"  Code-switching samples: {len(SAMPLE_CODE_SWITCH_TEXTS)}")

        total_lines = args.repeat_factor * (
            len(SAMPLE_TR_TEXTS) + len(SAMPLE_EN_TEXTS) + len(SAMPLE_CODE_SWITCH_TEXTS)
        )
        print(f"  Total training lines:  ~{total_lines:,}")

        corpus_iter = generate_corpus_iterator(
            SAMPLE_TR_TEXTS,
            SAMPLE_EN_TEXTS,
            SAMPLE_CODE_SWITCH_TEXTS,
            repeat_factor=args.repeat_factor,
        )
    else:
        print(f"  Turkish corpus files:       {args.tr_corpus}")
        print(f"  English corpus files:       {args.en_corpus}")
        print(f"  Code-switching corpus files: {args.cs_corpus}")
        corpus_iter = generate_corpus_from_files(
            args.tr_corpus,
            args.en_corpus,
            args.cs_corpus,
        )

    # Step 3: Train
    print("\n[3/4] Training Unigram model...")
    tokenizer = train_tokenizer(tokenizer, corpus_iter, vocab_size=args.vocab_size)
    print(f"  Training complete! Vocab size: {tokenizer.get_vocab_size():,}")

    # Step 4: Evaluate
    if not args.skip_eval:
        print("\n[4/4] Evaluating tokenizer...")
        evaluate_tokenizer(tokenizer)
    else:
        print("\n[4/4] Evaluation skipped.")

    # Save and optionally push
    save_and_push(
        tokenizer,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        hf_token=hf_token,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
