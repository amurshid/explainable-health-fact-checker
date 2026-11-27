"""Write the shared split ID files: shared/splits/{train,val,test}_ids.csv.

Run from the repo root:  python -m shared.make_splits
"""
from sklearn.model_selection import train_test_split

from shared.pubhealth import LABELS, SEED, SPLITS_DIR, clean, read_raw


def main():
    train_pool = clean(read_raw("train"))
    test = clean(read_raw("dev"))  # official dev split = our held-out test set

    # no claim may appear in both the training pool and the test set
    train_pool = train_pool[~train_pool["claim"].str.lower().isin(set(test["claim"].str.lower()))]

    train, val = train_test_split(train_pool, test_size=0.10, random_state=SEED, stratify=train_pool["label"])

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df[["claim_id", "label"]].to_csv(SPLITS_DIR / f"{name}_ids.csv", index=False)
        counts = df["label"].value_counts().reindex(LABELS).to_dict()
        print(f"{name:5s} {len(df):5d} claims  {counts}")

    claims = {n: set(d["claim"].str.lower()) for n, d in (("train", train), ("val", val), ("test", test))}
    assert not (claims["train"] & claims["val"]) and not (claims["train"] & claims["test"]) and not (claims["val"] & claims["test"])
    print(f"wrote {SPLITS_DIR}")


if __name__ == "__main__":
    main()
