#!/usr/bin/env python3
"""Prepare WikiText-2-raw test tokens using the model's fast tokenizer."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def wiki_text(archive, split="test"):
    if split not in ("test", "valid"):
        raise ValueError("only test and valid splits are supported")
    member = f"wikitext-2-raw/wiki.{split}.raw"
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read(member)
    with io.StringIO(raw.decode("utf-8"), newline=None) as stream:
        lines = [line if line.strip() else "" for line in stream]
    return "\n\n".join(lines), raw, len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="wikitext-2.zip")
    parser.add_argument("--tokenizer-json", required=True, type=Path,
                        help="tokenizer.json for the evaluated model")
    parser.add_argument("--output", default=Path("wikitext2-test.tokens"), type=Path)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    args = parser.parse_args()

    try:
        text, raw, lines = wiki_text(args.zip, args.split)
        try:
            from tokenizers import Tokenizer
        except ImportError:
            parser.exit(1, "prepare_wikitext2: install the official fast tokenizer "
                           "runtime with 'python3 -m pip install tokenizers'\n")
        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 2:
            raise ValueError("fewer than two tokens after tokenization")
        tokenizer_hash = hashlib.sha256(args.tokenizer_json.read_bytes()).hexdigest()
        args.output.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
        metadata = {
            "dataset": f"wikitext-2-raw-v1/{args.split}",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "tokenizer_sha256": tokenizer_hash,
            "text_construction": "WikiText loader: blank lines become empty; other lines retain trailing newline; join with two newlines; no BOS/EOS or chat template",
            "lines": lines,
            "tokens": len(ids),
            "scored_tokens": len(ids) - 1,
        }
        metadata_path = args.output.with_name(args.output.name + ".json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"prepare_wikitext2: {error}\n")
    print(f"{args.output}: {len(ids)} tokens, {len(ids) - 1} scored; metadata: {metadata_path}")


if __name__ == "__main__":
    main()
