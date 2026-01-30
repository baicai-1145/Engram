#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare SFT JSONL: prompt/response TSV -> {\"prompt\":...,\"response\":...}"
    )
    ap.add_argument("--input", required=True, help="Input TSV file: prompt<TAB>response per line")
    ap.add_argument("--output", required=True, help="Output JSONL file")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Bad TSV at line {line_no}: expected 2 columns, got {len(parts)}")
            prompt, response = parts
            fout.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()

