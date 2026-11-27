# Explainable Fact-Checking of Public Health Claims

EDS 6352 final project, Team 9. Given a health claim (e.g. *"Expired cake mix is toxic."*), the system:

1. **Retrieves evidence:** finds the matching fact-check article and picks its 3 most relevant sentences.
2. **Classifies veracity:** labels the claim `true`, `false`, `mixture` or `unproven`.
3. **Generates an explanation:** writes a short explanation from the claim and the 3 sentences.

Steps 2 and 3 both take the claim plus the 3 sentences from step 1 as input.

Dataset: [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking) (Kotonya and Toni, EMNLP 2020).

## Who did what

| Part | Owner | Folder | Model |
|---|---|---|---|
| Evidence retrieval | Ahnaf Murshid | [`retrieval/`](retrieval/) | Fine-tuned `all-MiniLM-L6-v2` bi-encoder |
| Veracity classification | Evandro L Silva | [`classification/`](classification/) | Fine-tuned `deberta-v3-base`, class-weighted loss |
| Explanation generation | Uzair Dastagir Mohammed | [`generation/`](generation/) | Fine-tuned `flan-t5-base` |
| Data cleaning and splits | All three | [`shared/`](shared/) | |

## Repository layout

```
├── shared/          data loading, cleaning and the train/val/test split ID files everyone uses
├── retrieval/       step 1
├── classification/  step 2
├── generation/      step 3
└── environment.yml  conda environment
```

## Setup

```bash
conda env create -f environment.yml
conda activate eds6352-final
```

Download `PUBHEALTH.zip` (see the dataset link above) and put it at `data/raw/PUBHEALTH.zip`. The `data/` folder is not tracked by git.

All commands below are run from the repo root.

## Shared splits

- Test set: the official `dev.tsv` (1,214 claims after cleaning).
- Train/val: the cleaned `train.tsv`, split 90/10, stratified by label, `random_state=42` (8,820 / 981 claims).
- Cleaning: HTML entities, tags and URLs removed; invalid labels, empty claims, short explanations (<3 words), short articles (<20 words) and duplicate claims dropped.

The split ID files are committed in [`shared/splits/`](shared/splits/). To regenerate them:

```bash
python -m shared.make_splits
```

To load a split in your own code:

```python
from shared.pubhealth import load_split
train = load_split("train")   # claim_id, claim, explanation, main_text, label
```
