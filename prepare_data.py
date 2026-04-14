#!/usr/bin/env python3
"""
Multrenizer - Data Preparation Pipeline
========================================
Downloads, cleans, and prepares a bilingual EN-TR corpus for tokenizer training.

Data sources:
  - Turkish:  Wikipedia TR  +  OPUS-100 TR side
  - English:  Wikipedia EN  +  OPUS-100 EN side
  - Code-switching: Synthetically generated from OPUS-100 parallel pairs

Collection targets:
  - Turkish-forward source collection (~65% TR / ~35% EN before training-time interleave)
  - Synthetic code-switching lines generated as a separate stream from OPUS pairs
  - Final training mix is defined in train_tokenizer.py

Output:
  data/
    tr_corpus.txt      - Turkish monolingual text (one sentence per line)
    en_corpus.txt      - English monolingual text (one sentence per line)
    cs_corpus.txt      - Code-switching synthetic text (one sentence per line)
    manifest.json      - Stats and metadata about the prepared corpus
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"

# How many sentences to collect from each source
# These are defaults -- override with CLI args
DEFAULT_TR_WIKI_LINES = 500_000
DEFAULT_EN_WIKI_LINES = 250_000
DEFAULT_OPUS_LINES = 100_000  # parallel pairs -> feeds both TR, EN, and CS

# Code-switching generation ratio from OPUS parallel pairs
CS_GENERATION_RATIO = 0.25  # 25% of OPUS pairs become code-switched


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

# Minimum line length (characters) to keep
MIN_LINE_LEN = 30
# Maximum line length to avoid garbage
MAX_LINE_LEN = 5000

# Patterns to filter out
RE_URL = re.compile(r"https?://\S+")
RE_MULTI_SPACE = re.compile(r"\s{2,}")
RE_REFERENCE = re.compile(r"\[\d+\]")
RE_WIKI_MARKUP = re.compile(r"\{\{[^}]*\}\}|\[\[[^]]*\]\]|<[^>]+>")
RE_SECTION_HEADER = re.compile(r"^=+\s.*\s=+$")


def clean_line(text: str) -> str | None:
    """Clean a single line of text. Returns None if the line should be skipped."""
    # Strip whitespace
    text = text.strip()
    if not text:
        return None

    # Skip section headers (== Title ==)
    if RE_SECTION_HEADER.match(text):
        return None

    # Remove wiki markup remnants
    text = RE_WIKI_MARKUP.sub("", text)

    # Remove references like [1], [23]
    text = RE_REFERENCE.sub("", text)

    # Remove URLs
    text = RE_URL.sub("", text)

    # Collapse multiple spaces
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    # Length filter
    if len(text) < MIN_LINE_LEN or len(text) > MAX_LINE_LEN:
        return None

    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Naively split a paragraph into sentences.
    Handles Turkish and English sentence boundaries.
    """
    # Split on sentence-ending punctuation followed by space + uppercase or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-zçğıöşü])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Wikipedia Downloader
# ---------------------------------------------------------------------------

def download_wikipedia(lang: str, max_lines: int, output_path: str):
    """
    Stream Wikipedia articles and extract clean sentences.

    Args:
        lang: Language code ('tr' or 'en').
        max_lines: Target number of clean lines to collect.
        output_path: Path to write the output text file.
    """
    config = f"20231101.{lang}"
    print(f"  Loading wikimedia/wikipedia ({config}) via streaming...")

    ds = load_dataset(
        "wikimedia/wikipedia", config,
        split="train",
        streaming=True,
    )

    collected = 0
    articles_processed = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for article in ds:
            if collected >= max_lines:
                break

            text = article.get("text", "")
            if not text:
                continue

            # Split article into paragraphs then sentences
            paragraphs = text.split("\n")
            for para in paragraphs:
                if collected >= max_lines:
                    break

                sentences = split_into_sentences(para)
                for sent in sentences:
                    if collected >= max_lines:
                        break

                    cleaned = clean_line(sent)
                    if cleaned:
                        f.write(cleaned + "\n")
                        collected += 1

            articles_processed += 1
            if articles_processed % 5000 == 0:
                print(f"    [{lang.upper()}] {articles_processed:,} articles -> "
                      f"{collected:,}/{max_lines:,} lines")

    print(f"    [{lang.upper()}] Done: {collected:,} lines from "
          f"{articles_processed:,} articles -> {output_path}")
    return collected


# ---------------------------------------------------------------------------
# OPUS-100 Downloader (Parallel EN-TR)
# ---------------------------------------------------------------------------

def download_opus(
    max_pairs: int,
    tr_output: str,
    en_output: str,
    cs_output: str,
    cs_ratio: float = CS_GENERATION_RATIO,
):
    """
    Stream OPUS-100 en-tr parallel corpus.
    Splits into:
      - TR monolingual sentences -> appended to tr_output
      - EN monolingual sentences -> appended to en_output
      - Synthetic code-switched sentences -> written to cs_output

    Args:
        max_pairs: Max parallel pairs to process.
        tr_output: Path to TR corpus file (append mode).
        en_output: Path to EN corpus file (append mode).
        cs_output: Path to code-switching corpus file (write mode).
        cs_ratio: Fraction of pairs to convert to code-switching.
    """
    print("  Loading Helsinki-NLP/opus-100 (en-tr) via streaming...")

    ds = load_dataset(
        "Helsinki-NLP/opus-100", "en-tr",
        split="train",
        streaming=True,
    )

    tr_count = 0
    en_count = 0
    cs_count = 0
    pairs_processed = 0

    with (
        open(tr_output, "a", encoding="utf-8") as f_tr,
        open(en_output, "a", encoding="utf-8") as f_en,
        open(cs_output, "w", encoding="utf-8") as f_cs,
    ):
        for item in ds:
            if pairs_processed >= max_pairs:
                break

            translation = item.get("translation", {})
            en_text = translation.get("en", "").strip()
            tr_text = translation.get("tr", "").strip()

            if not en_text or not tr_text:
                continue

            # Clean both sides
            en_clean = clean_line(en_text)
            tr_clean = clean_line(tr_text)

            pairs_processed += 1

            # Decide: monolingual or code-switch?
            if random.random() < cs_ratio and en_clean and tr_clean:
                # Generate code-switched variant
                cs_line = generate_code_switch(en_clean, tr_clean)
                if cs_line:
                    f_cs.write(cs_line + "\n")
                    cs_count += 1
            else:
                # Monolingual: write both sides to their respective files
                if tr_clean:
                    f_tr.write(tr_clean + "\n")
                    tr_count += 1
                if en_clean:
                    f_en.write(en_clean + "\n")
                    en_count += 1

            if pairs_processed % 20000 == 0:
                print(f"    [OPUS] {pairs_processed:,}/{max_pairs:,} pairs | "
                      f"TR:{tr_count:,} EN:{en_count:,} CS:{cs_count:,}")

    print(f"    [OPUS] Done: {pairs_processed:,} pairs -> "
          f"TR:{tr_count:,} EN:{en_count:,} CS:{cs_count:,}")
    return tr_count, en_count, cs_count


# ---------------------------------------------------------------------------
# Synthetic Code-Switching Generator
# ---------------------------------------------------------------------------

# Common TR suffixes that get attached to English words in plaza/dev speak
TR_SUFFIXES = [
    "'lamak", "'lemek", "'ladık", "'ledik", "'lıyoruz", "'liyoruz",
    "'layacağız", "'leyeceğiz", "'lanmış", "'lenmiş", "'landı", "'lendi",
    "'layan", "'leyen", "'lasın", "'lesin", "'ladım", "'ledim",
]

# Common English tech/business terms used in Turkish code-switching
EN_TERMS = [
    "deploy", "merge", "commit", "push", "pull", "review", "release",
    "sprint", "meeting", "deadline", "feedback", "budget", "target",
    "update", "debug", "refactor", "implement", "design", "feature",
    "bug", "fix", "test", "build", "scale", "optimize", "monitor",
    "track", "plan", "schedule", "report", "launch", "scope", "draft",
]

# TR sentence templates with EN word slots
CS_TEMPLATES = [
    "Bu {en_word}'ı {tr_verb} gerekiyor.",
    "{en_word} sürecinde bir sorun yaşadık.",
    "Yarınki {en_word}'e kadar bitirmemiz lazım.",
    "Son {en_word}'de konuştuğumuz gibi {tr_clause}.",
    "{en_word} yapmadan önce {tr_clause}.",
    "Bu {en_word}'ı tamamladıktan sonra {en_word2}'e geçelim.",
    "Takımla {en_word} yapacağız, {tr_clause}.",
    "{tr_clause}, sonra {en_word} yapalım.",
    "Bu sprint'te {en_word} ve {en_word2} var.",
    "{en_word} sonuçlarına göre {tr_clause}.",
]

TR_VERBS = [
    "tamamlamamız", "bitirmemiz", "kontrol etmemiz", "düzeltmemiz",
    "incelememiz", "hazırlamamız", "gözden geçirmemiz", "planlamamamız",
]

TR_CLAUSES = [
    "ekibi bilgilendirelim", "durumu değerlendirelim",
    "testleri çalıştıralım", "sonuçları paylaşalım",
    "kodu gözden geçirelim", "müşteriyi bilgilendirelim",
    "dokümantasyonu güncelleyelim", "gerekli düzenlemeleri yapalım",
    "planı revize edelim", "sistemi kontrol edelim",
]


def generate_code_switch(en_sent: str, tr_sent: str) -> str | None:
    """
    Generate a synthetic code-switched sentence from a parallel pair.

    Strategy 1: Word substitution -- replace some TR words with EN equivalents
    Strategy 2: Template-based -- fill templates with mixed content
    Strategy 3: Suffix attachment -- attach TR suffixes to EN words
    """
    strategy = random.choice(["substitute", "template", "suffix"])

    if strategy == "substitute":
        # Take TR sentence, swap ~30% of content words with the EN ones
        tr_words = tr_sent.split()
        en_words = en_sent.split()
        if len(tr_words) < 5 or len(en_words) < 3:
            return _template_fallback()

        # Pick random positions in TR to swap
        n_swaps = max(1, len(tr_words) // 4)
        swap_positions = random.sample(
            range(min(len(tr_words), len(en_words))),
            min(n_swaps, len(en_words)),
        )
        result = tr_words[:]
        for pos in swap_positions:
            if pos < len(en_words):
                result[pos] = en_words[pos]

        line = " ".join(result)
        return line if len(line) >= MIN_LINE_LEN else None

    elif strategy == "template":
        return _template_fallback()

    else:  # suffix
        # Attach TR suffixes to EN tech terms
        term = random.choice(EN_TERMS)
        suffix = random.choice(TR_SUFFIXES)
        clause = random.choice(TR_CLAUSES)
        templates = [
            f"Kodu {term}{suffix} için {clause}.",
            f"Önce {term}{suffix} sonra {clause}.",
            f"Bu haftaki hedefimiz {term}{suffix}.",
            f"Sistemi {term}{suffix} lazım, yoksa sorun çıkar.",
            f"{term.capitalize()}{suffix} tarafında {clause}.",
        ]
        line = random.choice(templates)
        return line if len(line) >= MIN_LINE_LEN else None


def _template_fallback() -> str:
    """Generate a code-switched sentence from templates."""
    template = random.choice(CS_TEMPLATES)
    en_word = random.choice(EN_TERMS)
    en_word2 = random.choice(EN_TERMS)
    tr_verb = random.choice(TR_VERBS)
    tr_clause = random.choice(TR_CLAUSES)

    line = template.format(
        en_word=en_word,
        en_word2=en_word2,
        tr_verb=tr_verb,
        tr_clause=tr_clause,
    )
    return line if len(line) >= MIN_LINE_LEN else None


# ---------------------------------------------------------------------------
# Extra Code-Switching: Devrik Syntax + Plaza Dili
# ---------------------------------------------------------------------------

def generate_extra_cs_lines(count: int) -> list[str]:
    """
    Generate additional code-switching lines to reach the target ratio.
    Includes devrik (inverted) syntax patterns common in spoken Turkish.
    """
    lines = []

    plaza_sentences = [
        "Deploy sürecinde unexpected bir error aldık, rollback yapmamız gerekiyor.",
        "Bu feature'ı implement ederken edge case'leri handle etmeyi unutmayalım.",
        "Sprint planning'de discuss ettiğimiz gibi, backend refactoring'i önceliklendirmeliyiz.",
        "Production'a push etmeden önce staging'de test edelim.",
        "Code review'da feedback aldım, design pattern'ı değiştirmemiz lazım.",
        "Bu API endpoint'i optimize etmemiz gerekiyor, response time çok yüksek.",
        "Database migration'ı run ettikten sonra cache'i invalidate etmeyi unutma.",
        "Kubernetes cluster'ında resource limit'leri adjust etmemiz lazım.",
        "CI/CD pipeline'ında build fail oluyor, dependency conflict var sanırım.",
        "Microservice architecture'a geçiş planını draft ettim, review eder misin?",
        "Meeting'i reschedule edelim, stakeholder'lar available değil bugün.",
        "Q3 target'larımızı exceed ettik ama Q4 forecast biraz conservative.",
        "Deadline'a kadar deliverable'ları finalize etmemiz gerekiyor.",
        "Budget approval'ı aldık, procurement process'i başlatabiliriz.",
        "Performance review'da feedback olarak leadership skill'lerimi geliştirmem söylendi.",
        "Bu PR'ı merge'lemeden önce conflict'leri resolve edelim.",
        "Hotfix branch'ini cherry-pick'leyip production'a deploy edeceğiz.",
        "Load testing sonuçlarına göre throughput'u improve etmemiz lazım.",
        "Standup'ta blocker'ları discuss edelim, sprint velocity düşüyor.",
        "Tech debt'i address etmek için refactoring sprint'i planlayalım.",
    ]

    devrik_sentences = [
        "Error unexpected bir deploy sürecinde aldık biz.",
        "Etmeyi handle edge case'leri unutmayalım implement ederken.",
        "Önceliklendirmeliyiz backend refactoring'i, planning'de discuss etmiştik.",
        "Push etmeden production'a, staging'de mutlaka test edilmeli.",
        "Gerekiyor optimize etmemiz endpoint'i, response time kabul edilemez.",
        "Unutma cache'i invalidate etmeyi, migration'dan sonra.",
        "Değil bugün available stakeholder'lar, reschedule edelim meeting'i.",
        "Söylendi feedback olarak skill'lerimi geliştirmem.",
        "Conservative biraz forecast, ama exceed ettik target'ları.",
        "Resolve edelim conflict'leri, merge'lemeden önce PR'ı.",
    ]

    social_media_cs = [
        "Bu outfit çok aesthetic olmuş, where did you get it?",
        "Vibe'ı çok iyi olan bir cafe keşfettim, literally her köşesi fotoğraflık.",
        "Bence bu trend overrated, insanlar sadece hype'a kapılıyor.",
        "Content creation işi sanıldığı kadar easy değil, çok effort gerektiriyor.",
        "Playlist'ime yeni şarkılar ekledim, shuffle'da çok güzel çalıyor.",
        "Bu filter çok flattering, skin tone'umu düzeltiyor resmen.",
        "Influencer'ların recommend ettiği ürünlerin çoğu overpriced bence.",
        "Notification'larımı kapattım, mental health'im için çok iyi oldu.",
        "Algoritma timeline'ımı bozuyor, relevant content göremiyorum.",
        "Bu podcast episode çok insightful, özellikle ikinci yarısı.",
    ]

    all_templates = plaza_sentences + devrik_sentences + social_media_cs

    for _ in range(count):
        # 60% from templates, 40% from synthetic generation
        if random.random() < 0.6:
            line = random.choice(all_templates)
            # Add slight variation
            if random.random() < 0.3:
                # Shuffle some words for devrik variant
                words = line.split()
                if len(words) > 4:
                    i, j = random.sample(range(len(words)), 2)
                    words[i], words[j] = words[j], words[i]
                    line = " ".join(words)
        else:
            # Generate from suffix attachment
            term = random.choice(EN_TERMS)
            suffix = random.choice(TR_SUFFIXES)
            clause = random.choice(TR_CLAUSES)
            templates = [
                f"Kodu {term}{suffix} için {clause}.",
                f"Önce {term}{suffix} sonra {clause}.",
                f"Bu haftaki hedefimiz {term}{suffix}.",
                f"Sistemi {term}{suffix} lazım, yoksa sorun çıkar.",
                f"{term.capitalize()}{suffix} tarafında {clause}.",
                f"Bugün {term}{suffix} işini halledelim.",
                f"Acil olarak {term}{suffix} lazım.",
            ]
            line = random.choice(templates)

        if line and len(line) >= MIN_LINE_LEN:
            lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Balance & Manifest
# ---------------------------------------------------------------------------

def count_lines(path: str) -> int:
    """Count lines in a text file."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_manifest(data_dir: str, tr_lines: int, en_lines: int, cs_lines: int):
    """Write a JSON manifest with corpus statistics."""
    total = tr_lines + en_lines + cs_lines
    manifest = {
        "corpus_stats": {
            "tr_lines": tr_lines,
            "en_lines": en_lines,
            "cs_lines": cs_lines,
            "total_lines": total,
            "tr_ratio": round(tr_lines / total, 4) if total else 0,
            "en_ratio": round(en_lines / total, 4) if total else 0,
            "cs_ratio": round(cs_lines / total, 4) if total else 0,
        },
        "files": {
            "tr_corpus": "tr_corpus.txt",
            "en_corpus": "en_corpus.txt",
            "cs_corpus": "cs_corpus.txt",
        },
        "target_ratios": {
            "tr": 0.65,
            "en": 0.35,
            "cs": 0.10,
        },
        "note": "CS lines are carved from the TR share. Effective TR mono = tr_ratio - cs_ratio.",
    }

    path = os.path.join(data_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written to {path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multrenizer: Download and prepare bilingual EN-TR corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (small dataset)
  python prepare_data.py --size small

  # Medium corpus (~500K lines)
  python prepare_data.py --size medium

  # Large corpus (~2M lines)
  python prepare_data.py --size large

  # Custom sizes
  python prepare_data.py --tr-wiki 1000000 --en-wiki 500000 --opus 200000
        """,
    )

    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default=None,
        help="Preset corpus size: small (~10K), medium (~500K), large (~2M)",
    )
    parser.add_argument("--tr-wiki", type=int, default=None, help="TR Wikipedia lines to collect")
    parser.add_argument("--en-wiki", type=int, default=None, help="EN Wikipedia lines to collect")
    parser.add_argument("--opus", type=int, default=None, help="OPUS-100 parallel pairs to process")
    parser.add_argument("--output-dir", type=str, default=DATA_DIR, help="Output directory (default: data/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-wiki-tr", action="store_true",
        help="Skip Wikipedia TR download (use existing file)",
    )
    parser.add_argument(
        "--skip-wiki-en", action="store_true",
        help="Skip Wikipedia EN download (use existing file)",
    )
    parser.add_argument(
        "--skip-opus", action="store_true",
        help="Skip OPUS-100 download (use existing file)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Resolve sizes
    presets = {
        "small":  {"tr_wiki": 5_000,   "en_wiki": 2_500,   "opus": 3_000},
        "medium": {"tr_wiki": 350_000, "en_wiki": 175_000, "opus": 100_000},
        "large":  {"tr_wiki": 1_200_000, "en_wiki": 600_000, "opus": 300_000},
    }

    if args.size:
        preset = presets[args.size]
        tr_wiki_target = args.tr_wiki or preset["tr_wiki"]
        en_wiki_target = args.en_wiki or preset["en_wiki"]
        opus_target = args.opus or preset["opus"]
    else:
        tr_wiki_target = args.tr_wiki or DEFAULT_TR_WIKI_LINES
        en_wiki_target = args.en_wiki or DEFAULT_EN_WIKI_LINES
        opus_target = args.opus or DEFAULT_OPUS_LINES

    data_dir = args.output_dir
    os.makedirs(data_dir, exist_ok=True)

    tr_path = os.path.join(data_dir, "tr_corpus.txt")
    en_path = os.path.join(data_dir, "en_corpus.txt")
    cs_path = os.path.join(data_dir, "cs_corpus.txt")

    print("=" * 70)
    print("  MULTRENIZER - Data Preparation Pipeline")
    print("=" * 70)
    print(f"  TR Wikipedia target:  {tr_wiki_target:>10,} lines")
    print(f"  EN Wikipedia target:  {en_wiki_target:>10,} lines")
    print(f"  OPUS-100 pairs:       {opus_target:>10,} pairs")
    print(f"  Output directory:     {data_dir}")
    print("=" * 70)

    # --- Step 1: Wikipedia TR ---
    print("\n[1/4] Wikipedia Turkish...")
    if args.skip_wiki_tr and os.path.exists(tr_path):
        tr_wiki_lines = count_lines(tr_path)
        print(f"  Skipped (existing: {tr_wiki_lines:,} lines)")
    else:
        # Write mode for wiki (first source)
        tr_wiki_lines = download_wikipedia("tr", tr_wiki_target, tr_path)

    # --- Step 2: Wikipedia EN ---
    print("\n[2/4] Wikipedia English...")
    if args.skip_wiki_en and os.path.exists(en_path):
        en_wiki_lines = count_lines(en_path)
        print(f"  Skipped (existing: {en_wiki_lines:,} lines)")
    else:
        en_wiki_lines = download_wikipedia("en", en_wiki_target, en_path)

    # --- Step 3: OPUS-100 (parallel en-tr) ---
    print("\n[3/4] OPUS-100 parallel corpus (en-tr)...")
    if args.skip_opus and os.path.exists(cs_path):
        opus_tr, opus_en, opus_cs = 0, 0, count_lines(cs_path)
        print(f"  Skipped (existing CS: {opus_cs:,} lines)")
    else:
        opus_tr, opus_en, opus_cs = download_opus(
            opus_target, tr_path, en_path, cs_path,
        )

    # --- Step 4: Balance check & extra CS generation ---
    print("\n[4/4] Balancing corpus ratios...")
    tr_total = count_lines(tr_path)
    en_total = count_lines(en_path)
    cs_total = count_lines(cs_path)
    grand_total = tr_total + en_total + cs_total

    print(f"  Current: TR={tr_total:,} ({tr_total/grand_total:.1%}), "
          f"EN={en_total:,} ({en_total/grand_total:.1%}), "
          f"CS={cs_total:,} ({cs_total/grand_total:.1%})")

    # Check if CS is below 10% target
    target_cs = int(grand_total * 0.10)
    if cs_total < target_cs:
        extra_needed = target_cs - cs_total
        print(f"  CS below 10% target. Generating {extra_needed:,} extra CS lines...")
        extra_lines = generate_extra_cs_lines(extra_needed)
        with open(cs_path, "a", encoding="utf-8") as f:
            for line in extra_lines:
                f.write(line + "\n")
        cs_total += len(extra_lines)
        grand_total = tr_total + en_total + cs_total

    print(f"\n  Final: TR={tr_total:,} ({tr_total/grand_total:.1%}), "
          f"EN={en_total:,} ({en_total/grand_total:.1%}), "
          f"CS={cs_total:,} ({cs_total/grand_total:.1%})")

    # --- Manifest ---
    write_manifest(data_dir, tr_total, en_total, cs_total)

    print("\nData preparation complete!")
    print(f"Run tokenizer training with:")
    print(f"  python train_tokenizer.py \\")
    print(f"    --tr-corpus {tr_path} \\")
    print(f"    --en-corpus {en_path} \\")
    print(f"    --cs-corpus {cs_path}")


if __name__ == "__main__":
    main()
