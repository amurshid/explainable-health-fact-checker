"""Load PUBHEALTH claims using the team's shared split manifest (shared/splits/split_manifest.jsonl).

The manifest lists every claim_id in train.tsv and dev.tsv with:
  disposition   train | validation | test | excluded
  reason        why a claim was excluded (e.g. holdout_overlap, unresolved_tsv_record)
  source_file, line_start, line_end, parse_method
                where the record sits in the TSV and how to parse it. Some fields contain quotes
                that make pandas' default reader merge rows, so records are read from their exact
                lines instead.

Test = claims from the official dev.tsv; train/validation come from train.tsv.

Text cleaning applied to model inputs: HTML entities decoded, HTML tags and URLs removed,
whitespace collapsed.
"""
import csv
import html
import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = REPO_ROOT / "data" / "raw" / "PUBHEALTH.zip"
MANIFEST = REPO_ROOT / "shared" / "splits" / "split_manifest.jsonl"

LABELS = ["false", "mixture", "true", "unproven"]
SPLITS = {"train": "train", "val": "validation", "test": "test"}  # our name -> manifest disposition
TSV_COLUMNS = ["claim_id", "claim", "date_published", "explanation", "fact_checkers",
               "main_text", "sources", "label", "subjects"]

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")


def clean_text(text):
    text = html.unescape(html.unescape(str(text)))  # twice: some rows are double-encoded (&amp;amp;)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def read_manifest():
    with open(MANIFEST, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


@lru_cache(maxsize=None)
def _tsv_lines(name):
    with zipfile.ZipFile(DATA_ZIP) as z:
        return z.read(name).decode("utf-8").split("\n")


def _parse(row):
    text = "\n".join(_tsv_lines(row["source_file"])[row["line_start"] - 1:row["line_end"]])
    if row["parse_method"] == "strict_quoted_tsv":
        fields = next(csv.reader(io.StringIO(text), delimiter="\t"))
    else:  # literal_tab_split: quotes are part of the text
        fields = text.split("\t")
    record = dict(zip(TSV_COLUMNS, fields))
    assert len(fields) == len(TSV_COLUMNS) and record["claim_id"].strip() == row["claim_id"], row["claim_id"]
    return record


@lru_cache(maxsize=None)
def load_records():
    """Every parseable manifest record (excluded ones too), with its split and cleaned text."""
    manifest = read_manifest()
    manifest = manifest[manifest["parse_method"].notna()]
    rows = []
    for _, m in manifest.iterrows():
        r = _parse(m)
        rows.append({"claim_id": m["claim_id"], "disposition": m["disposition"], "label": m["label"],
                     "claim": clean_text(r["claim"]), "explanation": clean_text(r["explanation"]),
                     "main_text": clean_text(r["main_text"])})
    return pd.DataFrame(rows)


def load_split(name):
    """Claims of one split ('train', 'val' or 'test'): claim_id, claim, explanation, main_text, label."""
    df = load_records()
    df = df[df["disposition"] == SPLITS[name]]
    return df.drop(columns="disposition").reset_index(drop=True).copy()


def read_test_tsv_articles():
    """Article texts of the released test.tsv (its claims have no usable labels; articles only)."""
    with zipfile.ZipFile(DATA_ZIP) as z:
        text = z.read("PUBHEALTH/test.tsv").decode("utf-8")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    col = rows[0].index("main_text")
    return [clean_text(r[col]) for r in rows[1:] if len(r) == len(rows[0])]


if __name__ == "__main__":
    for split in SPLITS:
        df = load_split(split)
        counts = df["label"].value_counts().reindex(LABELS).to_dict()
        print(f"{split:5s} {len(df):5d} claims  {counts}")
