import unittest
from tokenizers import Tokenizer

from train_tokenizer import (
    _PUNCTUATION_TOKENS,
    build_pre_tokenizer,
    build_turkish_normalizer,
)


class TrainTokenizerBehaviorTest(unittest.TestCase):
    def test_ascii_apostrophe_is_reserved_as_utility_token(self):
        self.assertIn("'", _PUNCTUATION_TOKENS)

    def test_normalizer_canonicalizes_curly_apostrophes(self):
        normalizer = build_turkish_normalizer()

        normalized = normalizer.normalize_str("İstanbul’da")

        self.assertEqual(normalized, "istanbul'da")

    def test_pre_tokenizer_preserves_ascii_apostrophe_as_own_token(self):
        pre_tokenizer = build_pre_tokenizer()

        pieces = [token for token, _ in pre_tokenizer.pre_tokenize_str("feature'ı")]

        self.assertEqual(pieces, ["feature", "'", "ı"])

    def test_pre_tokenizer_preserves_apostrophe_in_english_contractions(self):
        pre_tokenizer = build_pre_tokenizer()

        pieces = [token for token, _ in pre_tokenizer.pre_tokenize_str("can't")]

        self.assertEqual(pieces, ["can", "'", "t"])

    def test_shipped_tokenizer_artifact_preserves_apostrophes(self):
        tokenizer = Tokenizer.from_file("multrenizer-tokenizer/tokenizer.json")

        tokens = tokenizer.encode("feature'ı").tokens

        self.assertEqual(tokens, ["<s>", "feature", "'", "ı", "</s>"])


if __name__ == "__main__":
    unittest.main()
