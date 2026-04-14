import unittest

from benchmark import clean_display_token, encode_without_specials


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        if add_special_tokens:
            return [1, 10, 20, 2]
        return [10, 20]

class BenchmarkHelpersTest(unittest.TestCase):
    def test_encode_without_specials_prefers_special_free_path(self):
        tok = FakeTokenizer()

        ids = encode_without_specials(tok, "örnek")

        self.assertEqual(ids, [10, 20])
        self.assertEqual(tok.calls, [("örnek", False)])

    def test_clean_display_token_removes_common_prefix_artifacts(self):
        self.assertEqual(clean_display_token("Ġgüzel"), "güzel")
        self.assertEqual(clean_display_token("##lik"), "lik")
        self.assertEqual(clean_display_token("▁istanbul"), "istanbul")


if __name__ == "__main__":
    unittest.main()
