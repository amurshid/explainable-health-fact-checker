"""Load and clean PUBHEALTH, and read the team's shared train/val/test splits.

Cleaning rules (agreed in the proposal, same as the team's first pipeline notebook):
  - decode HTML entities, strip HTML tags and URLs, collapse whitespace
  - drop rows with a label outside {false, mixture, true, unproven}
  - drop empty claims, explanations under 3 words, articles under 20 words
  - drop duplicate claims (case-insensitive, keep the first) and duplicate claim IDs

Splits: official dev.tsv is our test set; cleaned train.tsv is split 90/10 into
train/val, stratified by label, random_state=42. Run `python -m shared.make_splits`
to write the ID files in shared/splits/.
"""
import html
import re
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = REPO_ROOT / "data" / "raw" / "PUBHEALTH.zip"
SPLITS_DIR = REPO_ROOT / "shared" / "splits"

LABELS = ["false", "mixture", "true", "unproven"]
SEED = 42

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")


def read_raw(name, zip_path=DATA_ZIP):
    """Read one of 'train', 'dev', 'test' straight from the PUBHEALTH zip."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"PUBHEALTH/{name}.tsv") as f:
            return pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)


def clean_text(text):
    text = html.unescape(html.unescape(str(text)))  # twice: some rows are double-encoded (&amp;amp;)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean(df):
    """Apply the cleaning rules; returns claim_id, claim, explanation, main_text, label."""
    df = df.copy()
    df["claim_id"] = df["claim_id"].astype(str).str.strip()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABELS)]
    for col in ("claim", "explanation", "main_text"):
        df[col] = df[col].map(clean_text)
    df = df[df["claim"].str.len() > 0]
    df = df[df["explanation"].str.split().str.len() >= 3]
    df = df[df["main_text"].str.split().str.len() >= 20]
    df = df[~df["claim"].str.lower().duplicated()]
    df = df[~df["claim_id"].duplicated()]
    return df[["claim_id", "claim", "explanation", "main_text", "label"]].reset_index(drop=True)


def load_split(name):
    """Cleaned rows of one split ('train', 'val' or 'test'), in the order of its ID file."""
    ids = pd.read_csv(SPLITS_DIR / f"{name}_ids.csv", dtype=str)
    source = clean(read_raw("dev" if name == "test" else "train"))
    out = source.set_index("claim_id").loc[ids["claim_id"]].reset_index()
    assert (out["label"].values == ids["label"].values).all(), f"{name}: labels differ from the ID file"
    return out
