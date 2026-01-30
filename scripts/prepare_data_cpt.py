#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare CPT JSONL: raw text -> {\"text\": ...}")
    ap.add_argument("--input", required=True, help="Input text file, one sample per line")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    ap.add_argument("--strip", action="store_true", help="Strip each line")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.strip:
                line = line.strip()
            if not line:
                continue
            fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()

