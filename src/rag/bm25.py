"""Portable BM25 helpers with conservative Arabic/English normalization."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ARABIC_MARKS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
TOKEN_RE = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u06FA-\u06FC]+|[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.I)
ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
DIGIT_MAP = str.maketrans({
    **{chr(0x0660 + index): str(index) for index in range(10)},
    **{chr(0x06F0 + index): str(index) for index in range(10)},
})

# The certified corpus is Arabic, while real banking questions frequently
# code-switch into English.  These aliases are query-only: the stored corpus
# bytes and release hashes remain unchanged.  Keep the map small and explicit
# so lexical expansion is reproducible and auditable.
QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "credit": ("ائتمان", "الائتمان", "ائتماني", "ائتمانية"),
    "registry": ("تسجيل", "سجل", "بيانات"),
    "check": ("استعلام", "فحص", "الاطلاع"),
    "loan": ("قرض", "قروض", "تسهيل", "تسهيلات"),
    "loans": ("قرض", "قروض", "تسهيل", "تسهيلات"),
    "limit": ("حد", "حدود", "السقف"),
    "limits": ("حد", "حدود", "السقف"),
    "msme": ("منشآت", "الصغيرة", "المتوسطة"),
    "cbe": ("المركزي", "المركزى"),
}


def normalize_for_bm25(text: str) -> str:
    value = unicodedata.normalize("NFC", text)
    value = ARABIC_MARKS.sub("", value).translate(ALEF_MAP).translate(DIGIT_MAP)
    return value.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_for_bm25(text))


def expand_query_tokens(query: str) -> list[str]:
    """Return normalized query terms plus explicit bilingual aliases."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_ALIASES.get(token, ()))
    return expanded


def build_bm25_records(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for chunk in chunks:
        records.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_version_id": chunk["document_version_id"],
            "tokens": tokenize(chunk.get("text_clean") or ""),
        })
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class BM25Index:
    """BM25Okapi-style scorer; corpus rows stay aligned to chunk records by ID."""

    def __init__(self, records: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
        self.records = records
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_terms = [Counter(row["tokens"]) for row in records]
        self.doc_lengths = [sum(terms.values()) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / len(records) if records else 0.0
        df = Counter(token for terms in self.doc_terms for token in terms)
        n = len(records)
        self.idf = {token: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = set(expand_query_tokens(query))
        if not terms or not self.records:
            return []
        results = []
        for idx, frequencies in enumerate(self.doc_terms):
            length_norm = 1.0 - self.b + self.b * (self.doc_lengths[idx] / self.avg_doc_length if self.avg_doc_length else 0.0)
            score = 0.0
            for term in terms:
                tf = frequencies.get(term, 0)
                if tf:
                    score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
            if score > 0:
                results.append((idx, score))
        results.sort(key=lambda item: (-item[1], self.records[item[0]]["chunk_id"]))
        return results[:top_k]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(record)
    return rows
