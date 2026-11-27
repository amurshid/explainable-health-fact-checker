"""Search the article corpus with a MiniLM bi-encoder and score how often it finds the right article.

The claim and the article texts are embedded separately (a bi-encoder), and articles are ranked
by cosine similarity to the claim. Two ways to represent an article:
  first     embed the article text; MiniLM keeps only its first 256 word pieces
  passages  embed each ~150-word passage; an article's score is its best passage's score

Metrics (proposal): top-1 accuracy = % of claims whose top-ranked article is the correct one;
Recall@5 / Recall@10 = % whose correct article is in the top 5 / 10. MRR@10 is reported as well.

Claims and articles can be preprocessed with one of the @handle/#hashtag options in shared/social.py.

Examples (from the repo root):
  python -m retrieval.evaluate --split val --repr passages
  python -m retrieval.evaluate --split val --repr first
  python -m retrieval.evaluate --split val --mode replace
  python -m retrieval.evaluate --model outputs/retrieval/models/<run> --split test
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from retrieval.corpus import OUT_DIR, load_claims, load_corpus
from shared.social import MODES, apply_social

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
TOP_K = 10
REPRS = ("first", "passages")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path=BASE_MODEL):
    model = SentenceTransformer(str(path), device=get_device())
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def encode(model, texts, batch_size=128):
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=len(texts) > 5000)


def article_units(corpus, repr_, mode="keep"):
    """The texts embedded for each article, and which article (row index) each text belongs to."""
    if repr_ == "first":
        texts, owner = corpus["text"].tolist(), np.arange(len(corpus))
    else:
        texts = [p for passages in corpus["passages"] for p in passages]
        owner = np.repeat(np.arange(len(corpus)), corpus["passages"].map(len).values)
    return [apply_social(t, mode) for t in texts], owner


def _cache_path(model_path, repr_, mode, corpus_fp):
    key = f"{model_path}|{repr_}|{mode}|{corpus_fp}"
    weights = Path(model_path) / "model.safetensors"
    if weights.exists():  # a retrained model at the same path must not reuse old embeddings
        key += f"|{weights.stat().st_mtime_ns}"
    return OUT_DIR / "cache" / f"emb_{repr_}_{mode}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pt"


def corpus_embeddings(model, model_path, corpus, repr_, mode="keep"):
    """Embeddings of every article unit; cached on disk per model, representation and social mode."""
    texts, owner = article_units(corpus, repr_, mode)
    corpus_fp = hashlib.sha1("|".join(corpus["article_id"]).encode()).hexdigest()[:12]  # a changed corpus never reuses old embeddings
    path = _cache_path(model_path, repr_, mode, corpus_fp)
    if path.exists():
        emb = torch.load(path)
    else:
        emb = encode(model, texts).cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, path)
    return emb.to(model.device), owner


@torch.no_grad()
def search(query_emb, unit_emb, unit_owner, n_articles, gold_idx=None, k=TOP_K, batch=256):
    """Top-k article indices and scores per query, plus the rank of each query's gold article."""
    device = unit_emb.device
    owner = torch.as_tensor(unit_owner, device=device)
    tops, scores, ranks = [], [], []
    for i in range(0, len(query_emb), batch):
        sims = query_emb[i:i + batch].to(device) @ unit_emb.T                  # [queries, units]
        art = torch.full((len(sims), n_articles), float("-inf"), device=device)
        art.scatter_reduce_(1, owner.unsqueeze(0).expand_as(sims), sims, reduce="amax")  # best unit per article
        sc, ix = art.topk(k, dim=1)
        tops.append(ix.cpu()); scores.append(sc.cpu())
        if gold_idx is not None:
            g = torch.as_tensor(gold_idx[i:i + batch], device=device).unsqueeze(1)
            # rank over all articles (ties with the gold article count against it); for analysis only
            ranks.append((art >= art.gather(1, g)).sum(1).cpu())
    ranks = torch.cat(ranks).numpy() if ranks else None
    return torch.cat(tops).numpy(), torch.cat(scores).numpy(), ranks


def retrieval_metrics(positions):
    """positions: 1-based position of the gold article in the returned list (inf if not in it)."""
    r = np.asarray(positions, dtype=float)
    return {"top1_accuracy": float((r == 1).mean()), "recall_at_5": float((r <= 5).mean()),
            "recall_at_10": float((r <= 10).mean()), "mrr_at_10": float(np.where(r <= 10, 1 / r, 0).mean()),
            "n_claims": int(len(r))}


def evaluate(model_path=BASE_MODEL, split="val", repr_="passages", mode="keep", model=None, corpus=None):
    """Rank all articles for every claim in a split; returns (metrics, per-claim predictions)."""
    corpus = load_corpus() if corpus is None else corpus
    model = load_model(model_path) if model is None else model
    claims = load_claims(split)
    unit_emb, owner = corpus_embeddings(model, model_path, corpus, repr_, mode)
    query_emb = encode(model, [apply_social(c, mode) for c in claims["claim"]])
    idx_of = {a: i for i, a in enumerate(corpus["article_id"])}
    gold = np.array([idx_of[a] for a in claims["gold_article_id"]])
    top, scores, ranks = search(query_emb, unit_emb, owner, len(corpus), gold_idx=gold)
    # metrics use the gold article's position in the returned top-10 list, i.e. exactly what is handed
    # downstream (score ties between near-identical articles are resolved as the search resolved them)
    position = np.array([list(row).index(g) + 1 if g in row else np.inf for row, g in zip(top, gold)])

    ids = corpus["article_id"].values
    preds = pd.DataFrame({"claim_id": claims["claim_id"], "gold_article_id": claims["gold_article_id"],
                          "top1_article_id": ids[top[:, 0]], "gold_rank": ranks,
                          "top10_article_ids": [json.dumps(ids[row].tolist()) for row in top],
                          "top10_scores": [json.dumps(np.round(row, 5).tolist()) for row in scores]})
    metrics = {"model": str(model_path), "split": split, "repr": repr_, "mode": mode, "n_articles": len(corpus),
               **retrieval_metrics(position), "median_gold_rank": float(np.median(ranks))}
    return metrics, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=BASE_MODEL, help="model name or local path (default: pretrained MiniLM)")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--repr", default="passages", choices=REPRS, dest="repr_")
    ap.add_argument("--mode", default="keep", choices=MODES, help="@handle/#hashtag option (shared/social.py)")
    ap.add_argument("--name", help="name for the result files (default: 'pretrained' or the model folder name)")
    args = ap.parse_args()

    name = args.name or ("pretrained" if args.model == BASE_MODEL else Path(args.model).name)
    metrics, preds = evaluate(args.model, args.split, args.repr_, args.mode)
    out = OUT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{args.split}_{args.repr_}_{args.mode}"
    (out / f"{stem}.json").write_text(json.dumps(metrics, indent=1))
    preds.to_csv(out / f"{stem}_predictions.csv", index=False)
    print(f"[{stem}] top-1 {metrics['top1_accuracy']:.2%} | R@5 {metrics['recall_at_5']:.2%} | "
          f"R@10 {metrics['recall_at_10']:.2%} | MRR@10 {metrics['mrr_at_10']:.4f} "
          f"({metrics['n_claims']} claims, {metrics['n_articles']} articles)")


if __name__ == "__main__":
    main()
