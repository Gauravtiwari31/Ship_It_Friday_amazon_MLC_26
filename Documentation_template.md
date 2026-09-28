# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Ship It Friday  
**Team Members:** Gaurav Tiwari (Team Leader), Harshi Singh, Tushar Singh Akhawat, Prajjwal Singh  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A scalable, record-centric blocking → learned-retrieval → cascaded-matching pipeline. Multi-view TF-IDF blocking is
complemented by a supervised multilingual bi-encoder that retrieves each Source-2/3 record's most likely Source-1 entity,
and a learned stage-1 filter shrinks the candidate set to **4.77 pairs per Source-1 entity**. Cross-fitted LightGBM stages
with competition/context features and three fine-tuned cross-encoders then decide one owner per record. France, which has
no training labels, is handled by an open-set variant of the same chain trained only on India + US labels.
Public macro F0.5 of our submission without the final 111-pair France re-add: **0.987133**.

---

## 2. Methodology

### 2.1 Problem Analysis
- Each S2/S3 record belongs to at most one Source-1 entity; many S1 entities have no match (an empty prediction scores 1),
  so false merges are expensive (F0.5 weights precision).
- Noise: abbreviations and legal-form variants (Pvt Ltd / Private Limited, SARL, LLC), word order, transliterations
  (Devanagari / Latin), typos, trade names ("doing business as"), missing or partial addresses, house-number variants
  (ranges, suffixes, truncation), state / postcode formatting.
- The test has about **twice the training density of unmatched S2/S3 records** (≈2.4 distractors per S1 vs 1.22), and
  France appears only in test. Both shift precision, so validation is done at the test's density and with held-out countries.
- Irreducible bucket: records with an empty address whose (sorted) core name is shared by ≥ 2 S1 entities of the country
  (the owner cannot be determined from the data).

### 2.2 Solution Strategy
**Approach Type:** Hybrid — blocking + learned dense retrieval + cascaded gradient-boosted classifiers + cross-encoders + one-owner decision.  
**Core Innovation:** (1) supervised bi-encoder retrieval as a candidate source *and* as features in every stage;
(2) a learned candidate filter (stage-1 probability ≥ 0.003) that makes `candidate_pairs.tsv` exactly the scored set
(4.77 / S1) without losing recall; (3) an open-set chain for countries without labels, validated leave-one-country-out
at the test's distractor density.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:** Unicode folding and transliteration (a character mapping learned from training pairs + anyascii
  fallback), legal-form and stop-word stripping into a *core name*, name variants (main / alternative / DBA / legal form),
  address tokens, words, house numbers, state; all country-agnostic.
- **Blocking keys used:** per country, hashed TF-IDF views over name keys, address keys, name×house-number cross keys and
  their combination; forward top-k per S1 (name 5, address 5, combined 15) and reverse top-2 per S2/S3 record; key
  document-frequency cap 6,000; cosine ≥ 0.02.
- **Targeted rescues:** transliteration, typo (edit-distance) and trade-name (DBA) candidates.
- **Learned retrieval:** multilingual-e5-small bi-encoder fine-tuned with InfoNCE (τ = 0.05, in-batch + mined hard
  negatives), two models cross-fitted by S1 fold (a record is never retrieved by the model that saw its pair); each
  S2/S3 record adds its top-1 S1 of the same country. Out-of-fold on train it added 672,135 pairs of which 61,693 true
  matches missed by lexical blocking.
- **Learned candidate filter:** a stage-1 LightGBM on pair features keeps pairs with probability ≥ 0.003 (≈22 → 4.8 pairs
  per S1 with unchanged downstream F0.5).
- **Candidate pairs generated:** 8,263,584 on test (**4.770 per Source-1 entity**; India+US 4.623, France 5.602).
- **How true matches were not lost:** multi-view forward + reverse top-k, rescues, dense retrieval, and a filter threshold
  chosen for near-lossless recall (out-of-fold). Every submitted match is contained in `candidate_pairs.tsv`.

---

## 4. Matching Model

**Features used:**
- Name features: RapidFuzz ratio / token-sort / token-set / partial / Jaro-Winkler on core, main, alternative and
  concatenated names; no-space variants; token Jaccard; legal-form agreement/conflict; first-token equality; name-key
  equality; name rarity counts (within the country); word-risk encodings of added/missing words (India/US only).
- Address features: token-set / sort / partial ratios, word overlap, house-number relations (equal, prefix/suffix,
  truncation, log-distance, same length), state agreement/conflict, address-key equality, emptiness, rarity.
- Other: blocking cosines per view; competition context (candidate counts and ranks within the S1 and within the record,
  margins to the best competitor); stage-1 probability context; same-source and sibling-address context; learned
  retriever features (is top-1 retrieval, cosine, margin to the runner-up); cross-encoder logits and their context.

**Model type:** LightGBM (cross-fitted by S1 fold): stage 1 (filter) → stage 2 (with recomputed contexts) → stage 3 on
uncertain pairs (0.01 < p < 0.99) with cross-encoder logits from three fine-tuned cross-encoders (e5-small raw view,
e5-small normalised view, e5-base raw view) → India+US edge meta model (lambdarank owner ranker + binary LightGBM on
group/competition features). Decision: one owner per S2/S3 record (highest probability), then thresholds.  
**France / open set:** the same chain trained only on India + US labels, without the country-specific word-risk features,
with the retriever features, chain = mean(stage 1, stage 2), stage 3 with the e5-small cross-encoder logit.  
**Threshold selection method:** India+US rank rule (best candidate of an S1 ≥ 0.50, other candidates ≥ 0.75), tuned
out-of-fold for F0.5. France: flat 0.95 (a precision-first threshold for a country without labels), plus an exact-key
re-add: a pair between 0.60/0.75 (rank rule) and 0.95 is accepted only when both the order-free core-name key and the
non-empty order-free address key are equal (leave-one-country-out precision 0.996 in both countries; 111 France pairs).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - India + US, out-of-fold on all 2.2M training S1: **0.991798** at training density; **0.991349** at the test's
    distractor density ("corrected 2x": every training distractor record cloned once, blocking/features/contexts recomputed
    with frozen models) — India 0.992119, US 0.990836; precision 0.99825, recall 0.97655.
  - France chain, leave-one-country-out (all stages and the retriever trained on one country, scored on the other at
    corrected 2x, 250k S1 per country): unseen US 0.98304, unseen India 0.96808; exact-key re-add on the same sample:
    unseen US 0.98283 → 0.98293 (277 added, 276 true), unseen India 0.96746 → 0.96792 (495 added, 493 true).
  - Public leaderboard without the re-add: 0.987133 (the final file adds 111 France matches; not separately scored).
- **Common false positives (wrong merges):** decoys — different branches/entities sharing a chain or generic name at
  nearby addresses; records whose house number differs only by a suffix or range; single-candidate S1s that should stay
  empty; in France, generic legal-form names.
- **Common false negatives (missed matches):** records with empty addresses and names shared by several S1 (ambiguous,
  irreducible); heavy abbreviation or trade names without address overlap; pairs outside all candidate sources.

---

## 6. Conclusion
Scalable candidate generation (lexical multi-view blocking + supervised dense retrieval + a learned filter) with
cross-fitted cascaded matchers gives 0.9913 out-of-fold at the test's density with under five candidates per entity.
Validating at the test's distractor density and leave-one-country-out was essential: the unlabelled country needs a
precision-first operating point.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/` core library (normalisation, blocking, rescues, features, LightGBM training,
contexts, cross-encoders, decision, evaluation, pass-1 driver `run_pipeline.py`, settings `config.py`); `src/final/` the
final system's steps (bi-encoder training and retrieval, pass 2 with retrieval candidates, corrected-2x pools, India+US
cascade + meta model, France open-set chain, `make_final.py`). `README.md` lists the exact commands from `dataset/` to
`output/matching_results.tsv` and `output/candidate_pairs.tsv`; `requirements.txt` pins versions.

### B. Additional Results
| system | candidates / S1 | India+US 1x | India+US corrected 2x |
|---|---|---|---|
| lexical blocking, full candidate set | ~21 | 0.98938 | 0.98867 |
| + learned candidate filter | 4.6 | 0.98937 | 0.98870 |
| + learned retrieval | 4.7 | 0.99165 | 0.99118 |
| + retriever features in all stages + edge meta model (final) | 4.6 | 0.99180 | 0.99135 |

### C. Compliance
Only the provided dataset. No external registries, business databases, geocoding APIs, commercial entity-resolution
services or external business data. Pretrained multilingual e5 checkpoints (MIT) fine-tuned on training pairs only. No
test labels, no pseudo-labels, no self-training, no leaderboard-driven per-example decisions; France models are trained
only on India + US labels, and all countries go through the same code.
