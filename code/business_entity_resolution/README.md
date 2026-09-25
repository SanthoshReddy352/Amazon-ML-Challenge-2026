# Business Entity Resolution: Amazon ML Challenge 2026

Pipeline: raw TSV → parquet → normalisation → blocking (candidate generation) → matching → `output/*.tsv`.
Only the provided training/test data is used. No external lookups, APIs or geocoding (see *Compliance*).

## Environment
```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.11
pip install -r requirements.txt
```
The pipeline runs on a 16 GB / 12-thread laptop. All heavy steps are chunked.

## Reproduce (run from this folder; paths assume the repo layout)
```bash
# 1. TSV -> parquet (13 s)
python -m src.data --raw-dir ../../student_resource/dataset --out-dir ../../artifacts/processed
# 2. Normalisation. Learns native-script transliteration maps from TRAIN pairs only, then normalises
#    all 6 files (~14 min)
python -m src.normalize --processed ../../artifacts/processed --out ../../artifacts/normalized --workers 6
# 3. Validation split (train S1 entities: fit / val / dev / leave-one-country-out)
python -m src.split --processed ../../artifacts/processed --out ../../artifacts/splits
# 4. Blocking: candidates per S1 (train: ~68 min for 2.2M S1; test similar)
python -m src.blocking --norm ../../artifacts/normalized --split-file ../../artifacts/splits/split_v1.parquet \
    --gt ../../artifacts/processed/train_gt_pairs.parquet --subset all --out ../../artifacts/blocking
python -m src.blocking --norm ../../artifacts/normalized --split-file ../../artifacts/splits/split_v1.parquet \
    --subset test --out ../../artifacts/blocking
# 5a. v0 baseline (exact rule) -> ../../output/matching_results.tsv + candidate_pairs.tsv
python -m src.baseline_rule --norm ../../artifacts/normalized --split test --out ../../output
# validate
cd ../../student_resource && python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

## Modules (`src/`)
| Module | Purpose |
|---|---|
| `data.py` | TSV → parquet; ground truth exploded to (s1_id, match_id) pairs |
| `dictionaries.py` | Hand-written language dictionaries: legal forms (US/IN/FR), street types, states, French department→region |
| `normalize.py` | Unicode/zero-width cleanup, transliteration, alias split (D.B.A./t/a), legal-form extraction, address parsing (state, house no., street, localities); learns the native-script→Latin token map from train pairs |
| `split.py` | Stratified folds over train S1; dev subset; leave-one-country-out subsets |
| `evaluate.py` | Exact macro F0.5 (singleton rule), blocking report (pair recall, oracle F0.5), TSV IO, experiment log |
| `blocking.py` | IDF-weighted token-overlap blocking (name words, skeletons, word pairs, concatenations, 5-grams, name×locality, house×street), top-K union of total/name/address rankings per source |
| `baseline_rule.py` | v0 exact-rule matcher (name key + house no. + state) |

Tests: `python -m pytest tests -q`

## Compliance
- Data: only the challenge's train/test files. The transliteration map is learned from the training pairs.
- Dictionaries in `dictionaries.py` are general language knowledge (abbreviations, state/region names), not business data.
- No external APIs, databases or geocoding.
