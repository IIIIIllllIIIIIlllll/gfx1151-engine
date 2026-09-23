#!/usr/bin/env python3
"""Protocol checks for the WikiText-2 raw test preparation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from prepare_wikitext2 import main, wiki_text


class WikiTextPreparationTest(unittest.TestCase):
    def test_line_join_and_test_split_only(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "wikitext-2.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("wikitext-2-raw/wiki.test.raw", " \n = Title = \n \n Body . \n")
                zipped.writestr("wikitext-2-raw/wiki.train.raw", "training text\n")
            text, raw, count = wiki_text(archive)
            self.assertEqual(text, "\n\n = Title = \n\n\n\n\n Body . \n")
            self.assertEqual(raw, b" \n = Title = \n \n Body . \n")
            self.assertEqual(count, 4)
            with self.assertRaises(ValueError):
                wiki_text(archive, "train")

    def test_metadata_and_no_special_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            archive = folder / "wikitext-2.zip"
            tokenizer_json = folder / "tokenizer.json"
            output = folder / "test.tokens"
            tokenizer_json.write_text("{}", encoding="utf-8")
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("wikitext-2-raw/wiki.test.raw", "first\nsecond\n")

            def encode(text, add_special_tokens):
                self.assertEqual(text, "first\n\n\nsecond\n")
                self.assertFalse(add_special_tokens)
                return SimpleNamespace(ids=[7, 8, 9])

            fake_tokenizer = SimpleNamespace(from_file=lambda path: SimpleNamespace(encode=encode))
            arguments = ["prepare_wikitext2.py", "--zip", str(archive),
                         "--tokenizer-json", str(tokenizer_json), "--output", str(output)]
            with patch.dict(sys.modules, {"tokenizers": SimpleNamespace(Tokenizer=fake_tokenizer)}), \
                 patch.object(sys, "argv", arguments):
                main()
            self.assertEqual(output.read_text(encoding="ascii"), "7\n8\n9\n")
            meta = json.loads((folder / "test.tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dataset"], "wikitext-2-raw-v1/test")
            self.assertEqual(meta["scored_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
