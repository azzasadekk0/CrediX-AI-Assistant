"""Normalize the certified text artifact line endings in-place.

The release manifest hashes canonical UTF-8/LF bytes. This one-shot utility is
kept in the RAG package so a Windows checkout can restore those exact bytes
before running the release validator.
"""
from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("src/rag_data/current/bm25_corpus.jsonl")
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if normalized != raw:
        path.write_bytes(normalized)
    print(f"normalized={normalized != raw} bytes={len(normalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
