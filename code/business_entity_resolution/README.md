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
# 5. (GPU, optional; run on Kaggle) multilingual-e5-small embeddings + exact top-6 cosine neighbours per S1/source
python -m src.embed_export --norm ../../artifacts/normalized --out ../../artifacts/kaggle_export   # upload as dataset
#    run kaggle/embed_knn/embed_knn.py on 2xT4 -> knn_train.parquet, knn_test.parquet (~1 h)
# 6. Union candidates = blocking + embedding neighbours
python -m src.knn --knn ../../artifacts/kaggle_out/knn_train.parquet --split train \
    --cands ../../artifacts/blocking/cands_all_df3000_t20n10a10 --out ../../artifacts/blocking/union_train
python -m src.knn --knn ../../artifacts/kaggle_out/knn_test.parquet --split test \
    --cands ../../artifacts/blocking/cands_test_df3000_t20n10a10 --out ../../artifacts/blocking/union_test
# 7. Pair features (subsets: fit_sample = stage-1 train, fit_sample2 = 600k fresh S1 for stage 2, dev, val, test)
for s in fit_sample dev val; do python -m src.features --split train --cands ../../artifacts/blocking/union_train \
    --subset $s --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/${s}_u; done
python -m src.features --split train --cands ../../artifacts/blocking/union_train --subset fit_sample2 --fit-sample 600000 \
    --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/fit_sample2_u
python -m src.features --split test --cands ../../artifacts/blocking/union_test \
    --knn ../../artifacts/kaggle_out/knn_test.parquet --out ../../artifacts/features/test_u
# 8. Stage 1 + stage-2 group consistency, val report + v3 scores for val/test (pair universe of the refiner)
python pipeline/run_v3.py --tag u --retrain-s1 --test        # -> models/lgb_{s1,v3}_u, test_scores_v3_u, val scores
# 9. Extra out-of-sample refiner data: every fit S1 unseen by stage 1/2 (fit_sample3 = 440k, fit_sample4 = the rest)
for n in 3 4; do python -m src.features --split train --cands ../../artifacts/blocking/union_train --subset fit_sample$n     --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/fit_sample${n}_u
  python pipeline/build_fit3.py --name $n; done                # -> refine/fit{3,4}_set.parquet
# 10. Stage-3 sibling-aware refiner (src/refine.py): 5-fold on val + fit3/fit4 as extra training rows; scores test
python pipeline/run_v4.py --tag v4f --folds 5 --extra refine/fit3_set.parquet refine/fit4_set.parquet --test
# 11. Blocking extension (src/blockext.py: address-code / name / street keys) scored by its own 5-fold LightGBM
python pipeline/run_v5.py --main-tag v4f --top 5 --folds 5 --test
# 12. Cross-encoder (GPU, Kaggle 2xT4): export uncertain pairs, fine-tune xlm-roberta-base, score val/test
python pipeline/ce_export.py                                  # -> artifacts/kaggle_ce/{train,val,test}.parquet (+ keys)
#     upload the three parquet files as a Kaggle dataset and run kaggle/ce_kernel/ce_train.py (~1h40)
#     download ce_val.parquet / ce_test.parquet into artifacts/kaggle_ce/
# 13. Blend cross-encoder + refiner/extension probabilities (5-fold on val) -> blended test scores
python pipeline/ce_blend.py --test                            # -> artifacts/test_scores_ce_{main,ext}.parquet
# 14. Decision layer (one owner per record, thr 0.75 main / 0.70 extension, empty-S1 rescue 0.5) + validator
python pipeline/make_submission.py --scores ../../artifacts/test_scores_ce_main.parquet     --ext ../../artifacts/test_scores_ce_ext.parquet --thr 0.75 --ext-thr 0.7 --empty-thr 0.5 --out ../../output
python pipeline/write_candidates.py --out ../../output/candidate_pairs.tsv
# 15. Validate
cd ../../student_resource && python utils/validate_submission.py --matching ../output/matching_results.tsv     --candidate ../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```
Scripts in `pipeline/` locate the data through the environment variable `AMLC_ROOT` (the folder that contains
`student_resource/` and `artifacts/`): `set AMLC_ROOT=C:\path\to\repo` (Windows) or `export AMLC_ROOT=/path/to/repo`.
(`src.baseline_rule` reproduces the v0 exact-rule baseline.)

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
| `embed_export.py` | Compact text export (normalised name + city) for the Kaggle embedding job |
| `knn.py` | Merges embedding neighbours: recall report, union candidate dirs, `knn_cos`/`knn_rank` features |
| `features.py` | ~65 pair features: name/address similarities, agreement flags, blocking scores, within-S1 and across-S1 competition, name-conflict and within-S1 name rank, embedding similarity |
| `train.py` | LightGBM training utilities, one-owner decision, threshold tuning |
| `stage2.py` | Group consistency: compares each candidate with the S1's best other candidates (stage-1 probabilities) |
| `decision.py` | Decision strategies (threshold, expected-F0.5); threshold won on validation |
| `predict.py` | v3-era test scoring + decision layer (kept for reproducing v3) |
| `refine.py` | Stage-3 features: postcode/apartment-aware house number with delta/edit distance (siblings vs typos), typo-tolerant name-token substitution + log-frequency bucket of the swapped word, acronym match, copy-support counts |
| `blockext.py` | Blocking extension: rare composite keys (house+street, name+street, house+name, Indian address code+state/locality, name key+state, adjacent address-word pairs), hashed and built per country |

`pipeline/`: driver scripts (`run_v3/v4/v5.py`, `build_fit3.py`, `ce_export.py`, `ce_blend.py`, `make_submission.py`,
`write_candidates.py`), plus `lbsim.py` (leaderboard simulator used for the French analysis).
`kaggle/`: GPU jobs (`embed_knn` for candidate retrieval, `ce_kernel` for the cross-encoder).

Models: a cascade of LightGBM binary classifiers (MIT licence) and a fine-tuned cross-encoder
`xlm-roberta-base` (MIT, 278M parameters). `intfloat/multilingual-e5-small` (MIT, 118M) retrieves candidates and
gives a similarity feature. All models are MIT-licensed and far below the 8B-parameter limit.

Tests: `python -m pytest tests -q`

## Compliance
- Data: only the challenge's train/test files. The transliteration map is learned from the training pairs.
- Dictionaries in `dictionaries.py` are general language knowledge (abbreviations, state/region names), not business data.
- No external APIs, databases or geocoding.
