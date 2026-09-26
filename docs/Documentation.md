# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Bloom
**Team Members:** G Santhosh Reddy, [member 2], [member 3]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve Source 2 and Source 3 records to Source 1 entities in four stages:
1. **Country-partitioned, multi-key blocking**, plus multilingual embedding neighbours and an address-code extension.
2. **A cascade of LightGBM matchers** over ~120 name, address, competition and group-consistency features.
3. **A multilingual cross-encoder** (xlm-roberta-base, MIT licence) fine-tuned on uncertain pairs and blended into the LightGBM probability.
4. **A decision layer** tuned for macro F0.5: one owner per record, thresholds, and a rescue step for S1 that would otherwise be empty.

The key innovations are:
- **Sibling-aware features** that separate the generator's hard negatives (same street, house number shifted by +1…+21, one name word changed) from true copies.
- **Blocking keys built on Indian address codes.**
- **The cross-encoder**, which fixed most of the French (unseen country) false merges.

Validation macro F0.5 is **0.9866** (US 0.9870, India 0.9862); the public leaderboard score is **0.9819**.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the full training data (`docs/eda_findings.md`):
- **Structure.** Every matched S2/S3 record belongs to exactly one S1, and matches always share `country`. Singletons are 5.6%, and the mean is 3.46 matches per S1. The per-S1 match-count distribution is identical for US and India, so the generator is country-independent.
- **Noise on true copies.**
  - Name changes: word permutations, OCR/typo edits, injected accents, legal-suffix swaps, appended descriptors, domain-style names and junk prefixes.
  - Aliases ("X D.B.A. Y", "fka"), acronyms ("JM" = Jerusalem Musique), whole-name rebrands to invented words, and native-script transliterations (9 Indic scripts).
  - Address changes: reordered components, abbreviations, injected `NULL`/`PMB`/`PO Box`, and dropped, typo'd or replaced house numbers.
- **Distractors (26% of train records, about 2× more per S1 on test).**
  - Each distractor is a *sibling* of an S1 entity: the name gains, swaps or loses one word (or changes legal form), and the house number is shifted by +1…+21 on the same street.
  - Distractors are one-offs: only 1.6% share name + house number with another distractor, versus 55% of true copies.
  - They almost never have an empty address (0.3%); empty-address records are 97.7% true matches.
- **Name collisions.** 49% of S1 share a normalised name with another S1, so the address must disambiguate.
- **France (test only, 15% of S1).**
  - Names come from a small vocabulary, so rarity-weighted scores are depressed: the name score is 82 vs 105 for US/India true matches.
  - 29% of French S1 share a name with another S1 in the same city.
  - S1 uses regions while S2/S3 use departments; street abbreviations are French (R., AV, ALL, BD).

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Blocking, then a cascade of gradient-boosted matchers plus a transformer cross-encoder, then a metric-aware decision layer.
**Core Innovation:**
1. Generator-aware features: house-number delta and edit distance (siblings vs typos), a postcode- and apartment-aware house-number parser, typo-tolerant name-token substitution with a frequency bucket for the swapped word, and "copy support" (how many other records repeat this record's name + house).
2. An address-code blocking extension for Indian records.
3. A multilingual cross-encoder blended only where the tree models are uncertain.
4. Honest validation: an S1-grouped split that keeps the full distractor pool, with out-of-fold stacking at every stage. A leaderboard simulator was built to reason about the unlabelled French data.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used**
Blocking is partitioned by country; `country` is treated as an open set.
- **Main blocker** (`src/blocking.py`): a rarity-weighted inverted index. Tokens with a document frequency above 3,000 are dropped.
  - Name words, consonant skeletons, word pairs, the concatenated name, and 5-grams for domain names.
  - Composite keys: name word + locality, house number + street word, house number + street skeleton, full name key.
  - Per S1 it keeps the top 20 overall, the top 10 by name and the top 10 by address.
- **Embedding neighbours** (`kaggle/embed_knn`): multilingual-e5-small (MIT) embeddings of "name, address"; exact top-6 cosine neighbours per S1 per source.
- **Address-code extension** (`src/blockext.py`, rare keys only, at most 50 pool records per key):
  - house number + street word
  - name word + street word
  - house number + name word
  - Indian address code (B-46, D-2/201, 1-8-506/2/A) + state
  - address code + locality word
  - exact name key + state
  - rare adjacent address-word pairs ("greenray apartments")

  New pairs are ranked by a cheap name + address similarity, the top 5 per S1 are kept, and the pairs are scored by a dedicated model.

**Candidate pairs generated:** test = 103,968,895 (blocking + embedding neighbours) + 6,942,258 (extension) ≈ **110.9M pairs** (~64 per S1). The full set is written to `output/candidate_pairs.tsv`.

**How you ensured true matches were not lost**
- Recall was measured on the validation S1 against the *full* pool:
  - blocking alone: 95.8%
  - plus embedding neighbours: 96.7%
  - plus the extension: **~98%**
- We analysed the misses by type and added keys for the recoverable ones. Widening every per-S1 cap to 50 adds only +0.4% recall for 2.3× the pairs, so it was rejected.

---

## 4. Matching Model

**Features used**
- **Name:** fuzzy ratio, partial, token-sort and token-set; Jaro-Winkler; concatenated-name ratio; skeleton token-set; alias-vs-core; name-key and core equality; token Jaccard; legal-form agreement and Jaccard; length differences.
  - Typo-tolerant counts of S1 words missing from the record and record words absent from S1, with the log-frequency bucket of the swapped word (referenced to train S1).
  - Acronym match; domain and Indic flags.
- **Address:** token-set and partial similarity of the full address, street and locality; state, house-number and postcode agreement; number-set Jaccard and overlap; empty flag.
  - Parsed house number (postcode- and apartment-aware) with equality, signed numeric delta, edit distance and a substring flag.
- **Blocking and competition:** blocking scores and ranks.
  - Within-S1 relative scores.
  - Across-S1 competition for the record (best owner, margin to the runner-up).
  - Copy-support counts: other records with the same name + house, and S1s sharing the name key.
- **Group consistency (stage 2):** similarity of each candidate to the S1's best other candidates, including their name, house number and address.
- **Embedding cosine** from multilingual-e5-small.

**Model type**
1. Stage 1: LightGBM, trained on 200k fit S1.
2. Stage 2: LightGBM with group-consistency features, trained on 600k fresh fit S1.
3. Stage 3 refiner: LightGBM with the sibling-aware features, trained on the validation folds plus ~1.0M further unseen fit S1 (3.1M pairs); 5-fold out-of-fold.
4. Extension model: LightGBM for extension pairs.
5. Cross-encoder: **xlm-roberta-base** (MIT, 278M parameters) fine-tuned on Kaggle (2×T4) on 930k (S1 text, record text) pairs from S1 disjoint from validation. It scores pairs whose LightGBM probability is in [0.02, 0.995).
6. Blend: a small LightGBM over (LightGBM logit, cross-encoder logit, pair flags), 5-fold on validation.

The cross-encoder lifts ranking quality (AUC) on the uncertain pairs from 0.943 to 0.973.

**Threshold selection method**
- One owner per record (each S2/S3 record is kept only for its highest-probability S1).
- Thresholds 0.75 (main) and 0.70 (extension), chosen by macro F0.5 on out-of-fold validation. They were stress-tested with distractor false positives doubled to mimic the test density, where the optimum stays at 0.75–0.80.
- S1 that would be empty get their best candidate if p ≥ 0.5.

---

## 5. Results & Error Analysis

- **F0.5 score (macro):**

  | Version | Validation (441k S1) | Public leaderboard |
  |---|---|---|
  | Rule baseline | 0.680 | 0.680 |
  | LightGBM v1 | 0.970 | 0.962 |
  | Refiner + extension (v5c) | 0.9821 | 0.9742 |
  | Plus the cross-encoder (v8) | **0.9866** | **0.9819** |

  US/India on test land ~0.0013 below validation. France, which has no labels, was estimated by decomposing leaderboard scores: ≈0.94 before the cross-encoder, ≈0.97 after.
- **Common false positives (wrong merges):**
  - Generator siblings: the same street and the name plus one word, with the house number shifted.
  - Identical names at adjacent house numbers.
  - In France, same-address one-word swaps between generic vocabulary names ("Tourcoing Foyer" vs "Tourcoing College"). The cross-encoder rejects these.
- **Common false negatives (missed matches):**
  - Empty-address copies of names shared by several S1s (true ties).
  - Copies that are both renamed (invented word) and have a corrupted house number.
  - Heavily truncated Indian addresses.
  - About 2% of true pairs never enter the candidate set.
  - About 62% of the remaining loss is "correct but missed a copy".

---

## 6. Conclusion
Understanding the data generator mattered more than model capacity. Features that separate siblings from true copies, and blocking keys for Indian address codes, each gave larger gains than hyperparameters. A multilingual cross-encoder applied only to the uncertain pairs added a further +0.0033 on validation and +0.0068 on the leaderboard, mostly by fixing French false merges.

The main lesson: with an unseen country, audit every feature for distribution shift. Two regressions came from count and frequency features that behave differently in France.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`:
  - `data.py`, `normalize.py`, `dictionaries.py`: parsing, normalisation, and transliteration maps learned from train.
  - `split.py`: S1-grouped folds.
  - `blocking.py`, `knn.py`, `blockext.py`: candidate generation.
  - `features.py`, `stage2.py`, `refine.py`: features.
  - `train.py`: LightGBM training and one-owner assignment.
  - `evaluate.py`: exact macro F0.5 and blocking metrics.
  - `predict.py`: submission writing.
- Driver scripts (`code/notebooks/`): `run_v3.py` (stage 1–2), `build_fit3.py` (extra refiner data), `run_v4.py` (refiner), `run_v5.py` (extension), `ce_export.py` + `code/kaggle/ce_kernel/ce_train.py` (cross-encoder on Kaggle), `ce_blend.py` / `ce_blend2.py` (blend), `make_submission.py` (decision layer + validator).
- The README gives the exact order and commands to regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
- **Model licences:** multilingual-e5-small (MIT, 118M parameters), xlm-roberta-base (MIT, 278M), LightGBM (MIT). All are within ≤ 8B parameters.
- **Data:** no external data, lookups or geocoding. Every map (transliteration, state and department tables) is derived from the training data or is a static dictionary shipped in `src/dictionaries.py`.
- **Ablation (validation F0.5):**

  | Version | F0.5 |
  |---|---|
  | v3 | 0.9748 |
  | + sibling-aware refiner | 0.9781 |
  | + blocking extension | 0.9798 |
  | + Indian address codes | 0.9821 |
  | + 1M extra unseen training S1 | 0.9833 |
  | + cross-encoder blend | 0.9866 |
