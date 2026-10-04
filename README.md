# Amazon ML Challenge 2026: Business Entity Resolution (Team Bloom)

For each S1 business record, find every matching S2/S3 record across noisy, multilingual (US / India / France)
name and address data. Metric: macro F0.5 per S1.

| Result | F0.5 |
|---|---|
| Final submission (v17), public leaderboard | **0.987408** |
| v17, full local validation | 0.99109 (US 0.99068, IN 0.99170) |

**Approach:** blocking (exact keys, rare tokens, TF-IDF char kNN, multilingual embedding kNN, reverse kNN) →
LightGBM pairwise matcher → sibling-aware refiner → fine-tuned multilingual cross-encoders (xlm-roberta-base,
mdeberta-v3-base) → band re-scorer → decision layer (one owner per record, calibrated thresholds).
Full write-up: [`docs/Documentation.md`](docs/Documentation.md), EDA: [`docs/eda_findings.md`](docs/eda_findings.md),
experiment log: [`docs/experiments.md`](docs/experiments.md), day-by-day progress: [`STATUS.md`](STATUS.md).

## Layout
| Path | Contents |
|---|---|
| `code/business_entity_resolution/src/` | Library modules (data, normalise, blocking, features, refine, train, …) + `tests/` |
| `notebooks/` | Pipeline drivers. The README in `code/business_entity_resolution/` calls these `pipeline/…` (that is where the submission zip puts them); in this repo run them from `notebooks/`, or set `AMLC_ROOT` to the repo root |
| `kaggle/` | GPU jobs run on Kaggle 2×T4: embedding kNN, cross-encoder training/scoring, bi-encoder |
| `infra/` | AWS helper (unused in the end; the Free plan had no EC2 quota) |
| `docs/`, `logs/`, `STATUS.md` | Documentation, run logs, progress tracker |

## Data
The challenge dataset is **not** included; its terms are the organisers'. Put the provided files in
`student_resource/dataset/` (`train/`, `test/` TSVs) before running anything.

## Pretrained artifacts (GitHub Release `v17-final`)
Everything git ignores but you need to skip the slow or GPU steps is attached to the
[`v17-final` release](https://github.com/SanthoshReddy352/Amazon-ML-Challenge-2026/releases/tag/v17-final).
Every zip except the submission stores paths relative to the repo root: **unzip it at the repo root** and the files
land where the scripts expect them. Unzip `Team_Bloom_submission.zip` somewhere else, since it holds its own code snapshot.

| Asset | Size | Restores | Replaces step (code README) |
|---|---|---|---|
| `lgb_models.zip` | 218 MB | `artifacts/models/`: every LightGBM model (stage 1/2, refiner folds, ext/ext2–5, band stack) | 8, 10, 11, 14–15, 17 training |
| `splits_eda.zip` | 11 MB | `artifacts/splits/` (fit/val/dev split, seeded) and `artifacts/eda/` | 3 |
| `ce1_xlmr_base_v1.zip` | 1.1 GB | Fine-tuned cross-encoder 1 (xlm-roberta-base) | 12 (`kaggle/ce_kernel`) |
| `ce2_xlmr_base_v2.zip` | 1.1 GB | Fine-tuned cross-encoder 2 (xlm-roberta-base, wider band) | 12 (`kaggle/ce_kernel2`) |
| `ce4_mdeberta_v3_base.zip` | 1.1 GB | Fine-tuned cross-encoder 4 (mdeberta-v3-base) | 12 (`kaggle/ce_kernel4`) |
| `biencoder_e5.zip` | 465 MB | Fine-tuned bi-encoder (multilingual-e5) | 15 ext4 (`kaggle/biencoder`) |
| `gpu_ce_scores.zip` | 276 MB | `artifacts/kaggle_ce*`, `kaggle_cefr`, `kaggle_ext2`: all cross-encoder val/test scores + pair keys | 12, 14, 15 Kaggle scoring |
| `gpu_knn_rknn_bfwd.zip` | 1.5 GB | `artifacts/kaggle_out`, `kaggle_rknn`, `kaggle_bfwd`: embedding neighbours (forward, reverse, bi-encoder forward) | 5, 15 ext3 |
| `gpu_bknn_bknn2.zip` | 1.9 GB | `artifacts/kaggle_bknn`, `kaggle_bknn2`: bi-encoder neighbours | 15 ext4 |
| `kaggle_export_text.zip` | 387 MB | `artifacts/kaggle_export`: the text export uploaded as the Kaggle dataset | 5 export |
| `Team_Bloom_submission.zip` | 958 MB | The exact final submission: code snapshot + `matching_results.tsv` + `candidate_pairs.tsv` | (final output) |

Checksums: `SHA256SUMS.txt` in the release.

Not included: the third cross-encoder's weights (`ce_kernel3`, mdeberta-v3-base) only ever lived on Kaggle;
its val/test scores are in `gpu_ce_scores.zip`. Intermediate CPU artifacts (`processed/`, `normalized/`,
`blocking/`, `features/`, `refine/`, …, about 80 GB) are left out; steps 1–4 and 6–11 regenerate them on a laptop.

## Reproduce
1. Python 3.11: `pip install -r code/business_entity_resolution/requirements.txt`
2. Add the dataset (see *Data*), then download and unzip the release assets at the repo root.
3. Follow [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) in order, skipping the
   GPU work (step 5, the training/scoring in step 12, and the Kaggle runs inside 14–15) because the release
   already holds those outputs. Step 17 writes the final `matching_results.tsv`.

To check the final output without rerunning anything, `Team_Bloom_submission.zip` holds the exact submitted
`matching_results.tsv` and `candidate_pairs.tsv` plus a snapshot of the code.
