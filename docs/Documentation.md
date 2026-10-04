# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Bloom
**Team Members:** G Santhosh Reddy, Pulipaka Sanjana, Krishnavamsi Manukinda
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve Source 2 and Source 3 records to Source 1 entities in four stages:
1. **Country-partitioned, multi-key blocking** in both directions:
   - S1-centric: each S1 keeps its best records.
   - Record-centric reverse blocking: each S2/S3 record keeps its best S1.
   - Also: multilingual embedding neighbours, an address-code extension, and a **bi-encoder retriever fine-tuned on our own training pairs** (multilingual-e5-small, exact top-k in both directions).
2. **A cascade of LightGBM matchers** over ~120 name, address, competition and group-consistency features.
3. **An ensemble of multilingual cross-encoders**, fine-tuned on uncertain pairs and blended into the LightGBM probability:
   - two xlm-roberta-base models (MIT licence);
   - mdeberta-v3-base models (MIT licence).
4. **A band re-scorer**: a LightGBM that re-scores every uncertain pair using the bi-encoder's view of the whole country (is this S1 the record's nearest S1? how far is the runner-up?), name-tie counts, copy support and the raw cross-encoder logits.
5. **A decision layer** tuned for macro F0.5:
   - one owner per record and thresholds;
   - a rescue step for S1 that would otherwise be empty;
   - a record-level model that resolves empty-address records between S1 that share a name.

The key innovations are:
- **Sibling-aware features** that separate the generator's hard negatives (same street, house number shifted by +1…+21, one name word changed) from true copies.
- **Blocking keys built on Indian address codes.**
- **The cross-encoders**, which fixed most of the French (unseen country) false merges.
- **Record-centric reverse blocking**: it recovers copies that the S1-centric caps push out of the candidate lists.
- **Tie-aware empty-address resolution.**
- **A fine-tuned bi-encoder used twice**: as a retriever for the copies that every lexical key missed (hard positives mined from our own blocking misses), and as global context for re-scoring uncertain pairs.

Validation macro F0.5 is **0.99109** (US 0.99068, India 0.99170) for the final pipeline (v17); its public leaderboard score is **0.987408** (our best; v15 scored 0.986685).

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

- **Record-centric reverse blocking** (`src/revblock.py`, "ext2"):
  - **Why it is needed.** The S1-centric caps drop a typo'd or rebranded copy whenever its S1 has many exact-name records elsewhere, even though the copy's own best S1 is the right one. On validation, 3.1% of true pairs never reached the matcher.
  - **How it works.** Every S2/S3 record queries an index of the S1 of its country, using the same typed tokens with IDF over S1 and max_df 300. It keeps its top 3 S1 overall, plus the top 1 by name and the top 1 by address. Records with an empty address keep their top 10.
  - **Features.** The record's best and second-best S1 score over *all* S1 are kept as features: relative score and the gap between the top two.
  - **Scoring.** New pairs get their own 77-feature LightGBM (5-fold out-of-fold on validation, AUC 0.997), then the cross-encoder blend.
  - **Result.** On validation: 3.09M new pairs, containing 8.3k true pairs (26% of the remaining blocking misses). On test: 15.3M new pairs.

- **Embedding retrieval over records** (ext3, `kaggle/embed_rknn`): every S2/S3 record's top-5 S1 by zero-shot multilingual-e5 cosine on "name, city". 6.0M new validation pairs but only 1.6k true (+0.00017).
- **Fine-tuned bi-encoder retriever** (ext4, `kaggle/biencoder`):
  - multilingual-e5-small fine-tuned on 1.2M (S1, copy) pairs from fit S1 only, including all 203k fit pairs our candidate generation missed (hard positives), with the strongest competing S1 as a hard negative. Text "name | address"; symmetric InfoNCE (τ 0.05).
  - Exact top-5 in both directions (record → S1 and S1 → record) per country.
  - Validation: 7.04M new pairs with 13.6k true (≈55% of the remaining blocking misses); own 72-feature LightGBM + the 4-encoder blend + tie guard: **+0.0016**.

**Candidate pairs generated:** test ≈ **191.9M pairs** (~111 per S1): blocking + embedding neighbours 104.0M, extension 6.9M, ext2 15.3M, ext3/ext4 embedding neighbours. The full set is written to `output/candidate_pairs.tsv`.

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

- **Empty-address rescue** (`notebooks/empty_rescue.py`):
  - **The problem.** 97.7% of empty-address records are true copies, yet 53% of them were missed. Mostly these are name ties: several S1 share the record's name, and the pair model splits its probability between them.
  - **The model.** A record-level LightGBM decides whether an unclaimed empty-address record goes to its top S1.
  - **Tie-break evidence.** It uses the final pair probability and evidence counted over **all** S1 of the country:
    - how many S1 share the record's name key;
    - how many share the key plus the record's legal form;
    - how many share the exact normalised name;
    - whether the top S1 shares the legal form or the normalised name.
  - **Result.** It accepts at q ≥ 0.75 (about 90% precise on out-of-fold validation).
- **Tie guard.** A cross-encoder sees one pair, not the other S1 that carry the same name. The ext2 blend was trained on a band that contains almost no tied empty-address records, yet on test it upgraded 2.9k French ones whose name is shared by 4+ S1. For empty-address records with a tied name key, ext2 therefore keeps the tie-aware LightGBM probability.

- **Band re-scorer** (`notebooks/band_stack.py`, v16):
  - **Scope.** Every pair of every kind whose final probability is in [0.01, 0.995) (validation 236k pairs, test 968k). Main/extension pairs with refiner p ≥ 0.999 were never cross-encoder-blended on validation; on test these are France's above-band pairs, so they are left unchanged.
  - **Features (49).** The final probability and kind; the fine-tuned bi-encoder's neighbour context over **all** S1 of the country (rank and cosine of the pair in the record → S1 and S1 → record top-k lists, best/2nd/5th cosines, gaps, "is the record's nearest S1"); the zero-shot e5 reverse neighbours; name-key tie counts over all S1; the 4 raw cross-encoder logits; address-equality flags; the S1's confident copies (pf ≥ 0.9) with how many share the record's name key, house number or address key; and the record's competing S1 (sum, max and count of the other S1's probabilities, whether this S1 is the record's best).
  - **Stability.** Only features that behave the same on validation and test are used. Competition over the *candidate* frame is left out, because on validation it only sees validation S1.
  - **Result.** 5-fold out-of-fold on validation: band AUC 0.914 → 0.930, F0.5 0.99016 → **0.99109** with the decision below. The bi-encoder context gives most of it (+0.00046).

**Threshold selection method**
- One owner per record (each S2/S3 record is kept only for its highest-probability S1).
- **Record-normalised decision** (v16c). A record has one owner, but pairs are scored independently, so a record wanted by several S1 gets a confident winner even when the runner-up is just as likely: on validation, contested winners at p 0.75–0.9 are true only 45%. The threshold and the empty-S1 rescue therefore use p' = p / max(1, sum of the record's p over all its S1). Validation understates this problem, because validation candidates contain only validation S1 (20% of S1). On test, contested winners at p ≥ 0.75 per 1k S1 are 2.2 (US), 4.0 (India) and 15.9 (France: generic names such as an address-less "Calais Loisirs" against 14 "Calais Loisirs …" S1), versus 0.8–1.2 on validation. Validation 0.99098 → 0.99105; kept on top of the contest-aware re-scorer (v17) as a guard for the many-way French ties.
- Thresholds 0.75 (main) and 0.70 (extension), chosen by macro F0.5 on out-of-fold validation. They were stress-tested with distractor false positives doubled to mimic the test density, where the optimum stays at 0.75–0.80.
- S1 that would be empty get their best candidate if p ≥ 0.5.
- ext2 pairs use threshold 0.75; the result is flat between 0.70 and 0.85.
- Expected-F0.5 subset selection per S1 was re-tested on the calibrated ensemble (and again, by Monte Carlo, on v16b). It gives the threshold rule's score (0.98703; 0.99099 vs 0.99098), so the simpler rule is kept. Calibration is exact on validation: 1,799 expected vs 1,806 actual false-positive pairs.

---

## 5. Results & Error Analysis

- **F0.5 score (macro):**

  | Version | Validation (441k S1) | Public leaderboard |
  |---|---|---|
  | Rule baseline | 0.680 | 0.680 |
  | LightGBM v1 | 0.970 | 0.962 |
  | Refiner + extension (v5c) | 0.9821 | 0.9742 |
  | Plus the cross-encoder (v8) | 0.9866 | 0.9819 |
  | 3–4 cross-encoders (v11) | 0.98703 | 0.983363 |
  | + reverse blocking + empty-address rescue (v13b) | 0.98836 | 0.984715 |
  | + bi-encoder retrieval (v15) | 0.99016 | 0.986685 |
  | + band re-scorer (v16b) | 0.99098 | — |
  | + record-normalised decision (v16c) | 0.99105 | — |
  | + contest features in the re-scorer (v17, final) | **0.99109** | **0.987408** |

  US/India on test land ~0.0013 below validation. France, which has no labels, was estimated by decomposing leaderboard scores: ≈0.94 before the cross-encoder, ≈0.97 after, ≈0.975 with v15.
- **Remaining validation loss (v16b, gain if each class were fixed alone):** identical-name empty-address ties 14.4k pairs (+0.0029, unresolvable from the data), blocking misses 11.3k (+0.0023), empty-address low-probability 7.8k (+0.0017), false positives 2.1k (+0.0013), non-empty low-probability 5.2k (+0.0011).
- A simulation of the test's ~1.9× distractor density on validation (cloning distractor records) costs only 0.00013 and never favours higher thresholds, so thresholds were left at their validation optimum.
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
- Driver scripts (`code/business_entity_resolution/pipeline/`, Kaggle jobs under `kaggle/`): `run_ext2_block.py` / `run_ext2.py` / `ext2_blend.py` (reverse blocking), `run_ext3.py` + `kaggle/embed_rknn`, `kaggle/biencoder` (ext3/ext4), `final_v13.py` (decision layer), `band_stack.py` (v16 re-scorer, final output). Earlier drivers: `run_v3.py` (stage 1–2), `build_fit3.py` (extra refiner data), `run_v4.py` (refiner), `run_v5.py` (extension), `ce_export.py` + `code/kaggle/ce_kernel/ce_train.py` (cross-encoder on Kaggle), `ce_blend.py` / `ce_blend2.py` (blend), `make_submission.py` (decision layer + validator).
- The README gives the exact order and commands to regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
- **Model licences:** multilingual-e5-small (MIT, 118M parameters; also fine-tuned as the bi-encoder), xlm-roberta-base (MIT, 278M), mdeberta-v3-base (MIT, 278M), LightGBM (MIT). All are within ≤ 8B parameters.
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
  | + 2nd xlm-r + mdeberta cross-encoders (v11) | 0.98703 |
  | + record-centric reverse blocking (ext2) | 0.98792 |
  | + empty-address rescue (v13) | 0.98824 |
  | + 4th cross-encoder (CE4, mdeberta 2 epochs) | 0.98836 |
  | + zero-shot e5 record neighbours (ext3) | 0.98853 |
  | + fine-tuned bi-encoder retrieval (ext4, v15) | 0.99016 |
  | + band re-scorer: bi-encoder context, ties, raw CE logits (v16) | 0.99077 |
  | + address equality and copy support in the re-scorer (v16b) | 0.99098 |
  | + record-normalised one-owner decision (v16c) | 0.99105 |
  | + the record's competing S1 probabilities as re-scorer features (v17) | 0.99109 |
  | (tested, not kept) bi-encoder S1 → record top-15 retrieval (ext5): 2.16M new pairs, 4.5k true, no gain | 0.99099 |
