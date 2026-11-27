"""The three options for Twitter handles (@name) and hashtags (#tag) that every stage tests.

  keep     text unchanged
  remove   @handles deleted; the '#' is dropped but the hashtag word is kept (#COVID19 -> COVID19),
           because hashtag words usually carry content
  replace  @handles become [Person1], [Person2], ... (the same handle gets the same number across
           all texts passed together, e.g. a claim and its evidence); '#' dropped as in 'remove'

Only model inputs are changed; gold labels and reference explanations never are.
Run `python -m shared.social` to see how each stage's tokenizer handles the three options.
"""
import re

MODES = ("keep", "remove", "replace")
HANDLE_RE = re.compile(r"(?<!\w)@\w{1,30}")   # not preceded by a letter, so emails (a@b.com) are untouched
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
_WS_RE = re.compile(r"\s+")


def apply_social_many(texts, mode):
    """Apply one option to several texts of one example, sharing the [PersonN] numbering."""
    if mode == "keep":
        return list(texts)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    people = {}

    def person(match):
        handle = match.group(0).lower()
        people.setdefault(handle, f"[Person{len(people) + 1}]")
        return people[handle]

    out = []
    for t in texts:
        t = HANDLE_RE.sub(" " if mode == "remove" else person, t)
        t = HASHTAG_RE.sub(r"\1", t)
        out.append(_WS_RE.sub(" ", t).strip())
    return out


def apply_social(text, mode):
    return apply_social_many([text], mode)[0]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    assert apply_social_many(["@CDCgov says #COVID19", "per @cdcgov and @WHO"], "replace") == \
        ["[Person1] says COVID19", "per [Person1] and [Person2]"]
    assert apply_social("mail me at a@b.com @x", "remove") == "mail me at a@b.com"

    example = "@CDCgov says #COVID19 vaccines are safe, per @WHO."
    tokenizers = {"MiniLM (retrieval)": "sentence-transformers/all-MiniLM-L6-v2",
                  "DeBERTa-v3 (classification)": "microsoft/deberta-v3-base",
                  "Flan-T5 (generation)": "google/flan-t5-base"}
    for label, name in tokenizers.items():
        tok = AutoTokenizer.from_pretrained(name)
        print(f"\n{label}")
        for mode in MODES:
            pieces = tok.tokenize(apply_social(example, mode))
            print(f"  {mode:8s} {len(pieces):2d} tokens: {' '.join(pieces)}")
