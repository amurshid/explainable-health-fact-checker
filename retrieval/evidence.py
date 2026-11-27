"""Pick the 3 evidence sentences for every claim and write the hand-off files for steps 2 and 3.

Uses the best fine-tuned model chosen on val by retrieval/train.py (outputs/retrieval/models/best.json).
For each claim, the sentences of an article are ranked by cosine similarity to the claim (same model,
same @handle/#hashtag option) and the top 3 are kept, best first.

Two versions are written, both one JSON line per claim, {"claim_id", "article_id", "sentences": [...]}:
  retrieval/evidence/retrieved_<split>.jsonl      sentences from the article the retriever ranks first
                                                  (the real end-to-end pipeline)
  retrieval/evidence/gold_article_<split>.jsonl   sentences from the claim's correct article
                                                  (the "known article" upper bound)
Sentences are the original article text; each stage applies its own @handle/#hashtag option.

Also scores the pretrained and the fine-tuned model on val and test (the only place the test set is
used) and writes retrieval/results/metrics.json and training_history.json.

Run from the repo root:  python -m retrieval.evidence
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retrieval.corpus import OUT_DIR, load_claims, load_corpus
from retrieval.evaluate import BASE_MODEL, encode, evaluate, load_model
from retrieval.train import MODELS_DIR
from shared.pubhealth import REPO_ROOT
from shared.social import MODES, apply_social

EVIDENCE_K = 3
HANDOFF_DIR = REPO_ROOT / "retrieval" / "evidence"
RESULTS_DIR = REPO_ROOT / "retrieval" / "results"
SPLITS = ("train", "val", "test")


def sentence_embeddings(model, model_path, corpus, mode):
    """Embeddings of every sentence of every article (cached), and each article's first-sentence offset."""
    flat = [apply_social(s, mode) for sents in corpus["sentences"] for s in sents]
    offsets = np.concatenate([[0], np.cumsum(corpus["sentences"].map(len).values)])
    corpus_fp = hashlib.sha1("|".join(corpus["article_id"]).encode()).hexdigest()[:12]
    weights = Path(model_path) / "model.safetensors"
    key = f"{model_path}|{mode}|{corpus_fp}|{weights.stat().st_mtime_ns if weights.exists() else ''}"
    path = OUT_DIR / "cache" / f"sentemb_{mode}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pt"
    if path.exists():
        emb = torch.load(path)
    else:
        emb = encode(model, flat).cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, path)
    return emb.to(model.device), offsets


def top_sentences(query_vec, art_idx, sent_emb, offsets, corpus, k=EVIDENCE_K):
    """The k sentences of one article most similar to the claim, best first, with their scores."""
    start, end = offsets[art_idx], offsets[art_idx + 1]
    scores, order = (sent_emb[start:end] @ query_vec).topk(min(k, end - start))
    sentences = corpus["sentences"].iloc[art_idx]
    return [sentences[i] for i in order.tolist()], [round(float(s), 5) for s in scores]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    best = json.loads((MODELS_DIR / "best.json").read_text())
    model_path, mode = best["path"], best["mode"]
    print(f"best model: {mode} epoch {best['epoch']} (val top-1 {best['val_top1_accuracy']:.2%})")
    corpus = load_corpus()
    model = load_model(model_path)
    sent_emb, offsets = sentence_embeddings(model, model_path, corpus, mode)
    idx_of = {a: i for i, a in enumerate(corpus["article_id"])}
    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)

    metrics = []
    for split in SPLITS:
        m, preds = evaluate(model_path, split, "passages", mode, model=model, corpus=corpus)
        claims = load_claims(split)
        assert (preds["claim_id"].values == claims["claim_id"].values).all()
        if split != "train":  # training claims were seen in training, so their scores say little
            metrics.append({**m, "model": "fine-tuned", "model_path": m["model"]})
            print(f"[fine-tuned/{split}] top-1 {m['top1_accuracy']:.2%} | R@5 {m['recall_at_5']:.2%} | R@10 {m['recall_at_10']:.2%}")

        queries = encode(model, [apply_social(c, mode) for c in claims["claim"]])
        details = {"claim_id": claims["claim_id"], "gold_article_id": claims["gold_article_id"],
                   "retrieved_article_id": preds["top1_article_id"],
                   "retrieved_is_correct": preds["top1_article_id"] == claims["gold_article_id"],
                   "gold_rank": preds["gold_rank"]}
        for cond, article_ids in (("retrieved", preds["top1_article_id"]), ("gold_article", claims["gold_article_id"])):
            records, scores = [], []
            for q, cid, aid in zip(queries, claims["claim_id"], article_ids):
                sents, sc = top_sentences(q, idx_of[aid], sent_emb, offsets, corpus)
                records.append({"claim_id": cid, "article_id": aid, "sentences": sents})
                scores.append(sc)
            write_jsonl(records, HANDOFF_DIR / f"{cond}_{split}.jsonl")
            details[f"{cond}_sentences"] = [json.dumps(r["sentences"], ensure_ascii=False) for r in records]
            details[f"{cond}_scores"] = [json.dumps(s) for s in scores]
            assert all(len(r["sentences"]) >= 1 for r in records)
        pd.DataFrame(details).to_csv(OUT_DIR / "results" / f"evidence_details_{split}.csv", index=False)
        print(f"[{split}] wrote {len(claims)} claims -> {HANDOFF_DIR}/{{retrieved,gold_article}}_{split}.jsonl")

    for split in ("val", "test"):
        m, _ = evaluate(BASE_MODEL, split, "passages", mode, corpus=corpus)
        metrics.append({**m, "model": "pretrained", "model_path": m["model"]})
        print(f"[pretrained/{split}] top-1 {m['top1_accuracy']:.2%} | R@5 {m['recall_at_5']:.2%} | R@10 {m['recall_at_10']:.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"selected": {k: best[k] for k in ("mode", "epoch")}, "evidence_k": EVIDENCE_K,
         "results": metrics}, indent=1))
    history = {m: json.loads((MODELS_DIR / m / "history.json").read_text())
               for m in MODES if (MODELS_DIR / m / "history.json").exists()}
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=1))
    print(f"wrote {RESULTS_DIR / 'metrics.json'} and training_history.json")


if __name__ == "__main__":
    main()
