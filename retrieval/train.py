"""Fine-tune MiniLM (all-MiniLM-L6-v2) to find a claim's fact-check article.

Training pairs: each training claim + one passage of its correct article, the passage the
pretrained model scores highest for that claim (the same max-over-passages scoring used in search).

Loss: multiple-negatives ranking loss. In a batch of 32 (claim, passage) pairs, the model has to
pick each claim's own passage out of all 32 passages: the other 31 act as wrong answers
("in-batch negatives"). It is a cross-entropy over the 32 cosine similarities (scaled by 20).
Batches never hold two claims of the same article, so no "wrong answer" is actually correct.

Only training claims are used for training. After every epoch the model is saved and scored
on the val set (top-1, Recall@5, Recall@10). The test set is not touched here.

Run from the repo root:
  python -m retrieval.train --mode keep        # one @handle/#hashtag option
  python -m retrieval.train --mode all         # keep, remove, replace, then pick the best
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from retrieval.corpus import OUT_DIR, load_claims, load_corpus
from retrieval.evaluate import BASE_MODEL, article_units, corpus_embeddings, encode, evaluate, load_model
from shared.social import MODES, apply_social

SEED = 42
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
SCALE = 20.0  # sentence-transformers' default for this loss
MODELS_DIR = OUT_DIR / "models"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def positive_passages(corpus, claims, mode):
    """For each claim: the passage of its correct article that the pretrained model ranks highest."""
    base = load_model()
    unit_emb, owner = corpus_embeddings(base, BASE_MODEL, corpus, "passages", mode)
    query_emb = encode(base, [apply_social(c, mode) for c in claims["claim"]])
    texts, _ = article_units(corpus, "passages", mode)
    first_unit = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=len(corpus)))])
    idx_of = {a: i for i, a in enumerate(corpus["article_id"])}
    positives = []
    for q, aid in zip(query_emb, claims["gold_article_id"]):
        a = idx_of[aid]
        start, end = first_unit[a], first_unit[a + 1]
        positives.append(texts[start + int((unit_emb[start:end] @ q).argmax())])
    del base
    return positives


def make_batches(article_ids, batch_size, rng):
    """Shuffle, then fill batches so that no batch holds two claims of the same article."""
    pending = list(rng.permutation(len(article_ids)))
    batches = []
    while pending:
        batch, seen, rest = [], set(), []
        for i in pending:
            if len(batch) < batch_size and article_ids[i] not in seen:
                batch.append(int(i))
                seen.add(article_ids[i])
            else:
                rest.append(i)
        if len(batch) < 2:  # a single leftover pair has no negatives
            break
        batches.append(batch)
        pending = rest
    return batches


def embed(model, texts):
    features = model.preprocess(texts)  # input_ids, attention_mask, ... plus a non-tensor 'modality' field
    features = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in features.items()}
    return F.normalize(model(features)["sentence_embedding"], dim=-1)


def train(mode, epochs=EPOCHS):
    """Fine-tune with one social mode; saves every epoch and returns the per-epoch history."""
    set_seed(SEED)
    corpus = load_corpus()
    claims = load_claims("train")
    run_dir = MODELS_DIR / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{mode}] scoring the pretrained model on val")
    base_metrics, _ = evaluate(BASE_MODEL, "val", "passages", mode, corpus=corpus)
    history = [{"epoch": 0, **{f"val_{k}": v for k, v in base_metrics.items() if k.startswith(("top1", "recall", "mrr"))}}]
    print(f"[{mode}] epoch 0 (pretrained): val top-1 {base_metrics['top1_accuracy']:.2%}")

    print(f"[{mode}] choosing a positive passage for each of {len(claims)} training claims")
    queries = [apply_social(c, mode) for c in claims["claim"]]
    positives = positive_passages(corpus, claims, mode)
    article_ids = claims["gold_article_id"].values
    batches = {e: make_batches(article_ids, BATCH_SIZE, np.random.default_rng(SEED + e)) for e in range(1, epochs + 1)}

    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = sum(len(b) for b in batches.values())
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in batches[epoch]:
            q = embed(model, [queries[i] for i in batch])
            p = embed(model, [positives[i] for i in batch])
            logits = SCALE * q @ p.T                          # [batch, batch] similarity matrix
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=logits.device))  # diagonal = correct pairs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        train_seconds = time.time() - t0

        ckpt = run_dir / f"epoch_{epoch}"
        model.save(str(ckpt))
        metrics, preds = evaluate(ckpt, "val", "passages", mode, model=model, corpus=corpus)
        preds.to_csv(ckpt / "val_predictions.csv", index=False)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_seconds": round(train_seconds, 1),
                  **{f"val_{k}": v for k, v in metrics.items() if k.startswith(("top1", "recall", "mrr"))}}
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        print(f"[{mode}] epoch {epoch}: loss {record['train_loss']:.4f} | val top-1 {record['val_top1_accuracy']:.2%} "
              f"| R@5 {record['val_recall_at_5']:.2%} | R@10 {record['val_recall_at_10']:.2%} "
              f"| train {train_seconds:.0f}s, total {time.time() - t0:.0f}s")
    return history


def pick_best():
    """Best (mode, epoch) by val top-1 over all trained modes (ties: higher MRR@10, then earlier epoch)."""
    rows = []
    for mode in MODES:
        path = MODELS_DIR / mode / "history.json"
        if path.exists():
            rows += [{"mode": mode, **r} for r in json.loads(path.read_text()) if r["epoch"] > 0]
    best = max(rows, key=lambda r: (r["val_top1_accuracy"], r["val_mrr_at_10"], -r["epoch"]))
    best["path"] = str(MODELS_DIR / best["mode"] / f"epoch_{best['epoch']}")
    (MODELS_DIR / "best.json").write_text(json.dumps(best, indent=1))
    print("\nval top-1 by mode and epoch:")
    for r in rows:
        print(f"  {r['mode']:8s} epoch {r['epoch']}: {r['val_top1_accuracy']:.2%}")
    print(f"best: {best['mode']} epoch {best['epoch']} ({best['val_top1_accuracy']:.2%}) -> {MODELS_DIR / 'best.json'}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="all", choices=(*MODES, "all"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    for mode in (MODES if args.mode == "all" else (args.mode,)):
        train(mode, args.epochs)
    pick_best()


if __name__ == "__main__":
    main()
