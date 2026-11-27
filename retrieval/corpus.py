"""The article collection that retrieval searches over.

Every unique PUBHEALTH article becomes one candidate: the articles of our train, val
and test claims, plus the article texts of the released test.tsv (its labels are not
used; those articles only act as extra candidates, so the search covers all ~12,300
claims' articles as the proposal says). Identical texts are merged into one article.

Each article is also split into sentences (used to pick the 3 evidence sentences)
and into passages of whole sentences of up to ~150 words (used for article search,
because MiniLM only reads the first 256 word pieces of a text).
"""
import hashlib
import pickle

import nltk
import pandas as pd

from shared.pubhealth import REPO_ROOT, clean_text, load_split, read_raw

OUT_DIR = REPO_ROOT / "outputs" / "retrieval"
CORPUS_CACHE = OUT_DIR / "corpus.pkl"

MIN_SENTENCE_WORDS = 5    # shorter "sentences" are usually captions or leftovers
MAX_SENTENCE_WORDS = 80   # longer ones (missing punctuation) are cut into 60-word pieces
PASSAGE_WORDS = 150       # ~200 word pieces, under MiniLM's 256 limit


def article_id(text):
    """Stable ID for an article text; same text (ignoring case and spacing) -> same ID."""
    norm = " ".join(text.lower().split())
    return "A" + hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


def split_sentences(text):
    out = []
    for sent in nltk.sent_tokenize(text):
        words = sent.split()
        if len(words) < MIN_SENTENCE_WORDS:
            continue
        if len(words) > MAX_SENTENCE_WORDS:
            pieces = [" ".join(words[i:i + 60]) for i in range(0, len(words), 60)]
            out += [p for p in pieces if len(p.split()) >= MIN_SENTENCE_WORDS]
        else:
            out.append(sent.strip())
    return out or [" ".join(text.split()[:60])]


def make_passages(sentences):
    """Group consecutive sentences into passages of at most PASSAGE_WORDS words."""
    passages, current, n_words = [], [], 0
    for sent in sentences:
        n = len(sent.split())
        if current and n_words + n > PASSAGE_WORDS:
            passages.append(" ".join(current))
            current, n_words = [], 0
        current.append(sent)
        n_words += n
    if current:
        passages.append(" ".join(current))
    return passages


def build_corpus():
    texts = []
    for split in ("train", "val", "test"):
        texts += load_split(split)["main_text"].tolist()
    extra = read_raw("test")["main_text"].map(clean_text)
    texts += [t for t in extra if len(t.split()) >= 20]

    corpus = pd.DataFrame({"article_id": [article_id(t) for t in texts], "text": texts})
    corpus = corpus.drop_duplicates("article_id").sort_values("article_id").reset_index(drop=True)
    corpus["sentences"] = corpus["text"].map(split_sentences)
    corpus["passages"] = corpus["sentences"].map(make_passages)
    return corpus


def load_corpus():
    """The corpus DataFrame (article_id, text, sentences, passages), cached after the first build."""
    if CORPUS_CACHE.exists():
        with open(CORPUS_CACHE, "rb") as f:
            return pickle.load(f)
    corpus = build_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_CACHE, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


def load_claims(split):
    """One split's claims with the ID of their correct (gold) article."""
    df = load_split(split)
    df["gold_article_id"] = df["main_text"].map(article_id)
    return df


if __name__ == "__main__":
    corpus = load_corpus()
    n_pass = corpus["passages"].map(len)
    print(f"{len(corpus)} articles | {corpus['sentences'].map(len).sum()} sentences | "
          f"{n_pass.sum()} passages ({n_pass.mean():.1f} per article)")
    for split in ("train", "val", "test"):
        claims = load_claims(split)
        assert claims["gold_article_id"].isin(set(corpus["article_id"])).all()
        print(f"{split}: {len(claims)} claims, all gold articles in the corpus")
