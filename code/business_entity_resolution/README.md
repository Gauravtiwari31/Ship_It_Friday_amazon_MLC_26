# Ship It Friday — Business Entity Resolution: final solution (27 Sep 2026)

`output/matching_results.tsv` + `output/candidate_pairs.tsv` of this package are the files of our final leaderboard
submission, sha256 `22b4ba58c4c0c187e6f8480e06f1c0d85adb30412f28cac6896ca5595906f3dd` and
`67fdf4810f457333083d8e07b74754caba8154ba208b8a4084ae4e016bfc1ff7` (= our 0.987133 public submission + 111 France
matches from the exact-key re-add rule of step 8).
1,732,544 test Source-1 rows; 8,263,584 candidate pairs (**4.770 per Source-1 entity**); 5,860,665 matches.

**Data and compliance.** Only the provided `dataset/` files. No external registries, business databases, geocoding
APIs, commercial entity-resolution services or other external data. Pretrained language models (MIT licence,
`intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`, from the Hugging Face hub) are fine-tuned on the
provided training pairs only. **No model parameter or threshold is fitted on test data**: no pseudo-labels, no
self-training (`config.UNSEEN_SELFTRAIN = None`; the self-training code is not part of this package). Test records are
only normalised, blocked, retrieved and scored by frozen models. **France has no training labels: every France model is
trained only on the provided India + US training labels** (the "unseen-country" chain), and all countries are processed
by the same code (no country-specific rules or dictionaries).

## Environment
Python 3.14.2 venv (`python -m venv .venv`, then `pip install -r code/business_entity_resolution/requirements.txt`;
torch needs its CUDA build, e.g. `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`).
Hardware used: Windows 11, 16 CPU threads, 64 GB RAM, one NVIDIA RTX 4090 laptop GPU (16 GB). Run one GPU job at a time.
Layout expected (run every command from the folder that contains `code/`, `dataset/` and, optionally, `utils/` with the
official validator):
```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv   (as provided)
dataset/test/{test_source1,test_source2,test_source3}.tsv
code/business_entity_resolution/src/...        (this package)
```
Intermediate folders (`work_v17r`, `work_v17r_2x`, `work_lg`, `work_lg_2x`, `work/...`, `submission/...`) are created
by the scripts. `P=python` below means the venv interpreter.

## Code map
- `src/` — core library: `prepare.py`/`normalize.py`/`run_prepare.py` (normalisation, name/address parsing,
  transliteration learned from training pairs), `blocking.py`/`rescue.py`/`run_blocking.py` (multi-view TF-IDF top-k
  blocking + targeted rescues + learned-retrieval candidates), `features*.py`/`run_features.py` (pair features),
  `train.py` (cross-fitted LightGBM), `stage2.py` (context features), `cross_encoder.py` (e5 cross-encoders),
  `decide.py` (one owner per record + thresholds), `evaluate.py` (macro F0.5), `run_pipeline.py` (pass-1 driver),
  `config.py` (all settings).
- `src/final/` — the final system's steps, in run order below.

## Run order (exact commands)
1. **Pass 1** — normalisation, lexical blocking + rescues, features, stage-1/2 LightGBM, the two e5-small cross-encoders
   (raw and normalised record views; cross-fitted by Source-1 fold) and stage 3. Produces `work_v17r` (features, CE fold
   models `work_v17r/ce_model_v9`, `work_v17r/ce_model_v9_norm` and their out-of-fold scores). Approx. 5-7 h (GPU for the CEs).
   ```
   P -u code/business_entity_resolution/src/run_pipeline.py --data dataset --work work_v17r --out output_pass1
   ```
2. **Corrected-2x validation pool for pass 1** (the test has about twice the training density of unmatched S2/S3
   records; every training distractor record is cloned once, then blocking, features and contexts are recomputed with
   frozen models). Needed by steps 3-4. Approx. 1-2 h.
   ```
   P -u code/business_entity_resolution/src/final/sim_density.py work_v17r work_v17r_2x dataset --factor 2 --detwin
   ```
3. **Third cross-encoder (e5-base, raw view)**, cross-fitted, and its stage-3 integration → `work/audit/099/ce_e5base`,
   `work/audit/099/s3ce3`. Approx. 3-4 h GPU.
   ```
   P -u code/business_entity_resolution/src/final/s3_ce3.py --steps train,oof,2x,test
   ```
4. **Hard negatives for retrieval training** (out-of-fold pass-1 scores) → `work/audit/ceiling6h/retrieval/champion_oof_1x.parquet`.
   ```
   P -u code/business_entity_resolution/src/final/retrieval_audit.py
   ```
5. **Learned retrieval** — supervised e5-small bi-encoder ("query: name | address", mean pooling, InfoNCE τ = 0.05,
   in-batch + mined hard negatives), two models cross-fitted by Source-1 fold; then every S2/S3 record's top-1 Source-1
   of its country (out-of-fold on train; model A on test) → `work/audit/leader_gap/bienc_cands_{train,test}.parquet`.
   Approx. 15 min training per model + 1 h retrieval (GPU).
   ```
   P -u code/business_entity_resolution/src/final/biencoder.py --fold 0 --n-train 600000 --out work/audit/leader_gap/bienc_small_f0
   P -u code/business_entity_resolution/src/final/biencoder.py --fold 1 --n-train 300000 --no-eval --out work/audit/leader_gap/bienc_small_f1
   P -u code/business_entity_resolution/src/final/retrieve_cands.py
   ```
6. **Pass 2 with retrieval candidates** (`RESCUE += bienc`, fresh folder `work_lg`) and its corrected-2x pool
   (`work_lg_2x`, validation only). Approx. 3 h + 1.5 h.
   ```
   P -u code/business_entity_resolution/src/final/lg_pipeline.py work_lg
   P -u code/business_entity_resolution/src/final/lg_sim2x.py 1
   P -u code/business_entity_resolution/src/final/lg_sim2x.py 2
   ```
7. **India + US** — compact cascade with retriever features in all stages: stage-1 LightGBM → keep p1 ≥ 0.003 (this is
   the candidate set) → contexts recomputed on the kept pairs → stage 2 → stage 3 on 0.01 < p < 0.99 with the three
   cross-encoder logits (missing logits are scored with the fold models) → edge meta model (lambdarank owner ranker +
   binary LightGBM, cross-fitted) → one owner per record, rank rule 0.50 / 0.75 → India+US rows. Approx. 2 h.
   ```
   P -u code/business_entity_resolution/src/final/iu_cascade_bi.py --work work_lg --work2 work_lg_2x --out work/iu_claude/iu_cascade_bi
   P -u code/business_entity_resolution/src/final/owner_rank3.py
   P -u code/business_entity_resolution/src/final/iu_export.py work/iu_claude/owner_rank3/test_final_iu_meta.parquet 0.50 0.75 submission/iu_claude work/iu_claude/final_test_scores.parquet
   ```
   → `submission/iu_claude/matching_results_iu.tsv`, `submission/iu_claude/candidate_pairs_iu.tsv`
8. **France (open set; trained on India + US labels only)** — the same chain without the country-specific word-risk
   features, with retriever features: stage 1 (retrained) → p1 ≥ 0.003 → contexts → stage 2 → chain = mean(stage 1,
   stage 2) → stage 3 on the chain's zone with the e5-small CE logit → one owner per record, flat threshold 0.95.
   Approx. 1 h.
   ```
   P -u code/business_entity_resolution/src/final/fr_cascade_bi.py --work work_lg --out work/audit/leader_gap/lg_frcascade_bi
   P -u code/business_entity_resolution/src/final/fr_meta_prod.py --top 0.95 --rest 0.95 --tag bi_flat95 --no-meta
   ```
   → `submission/fr_claude/matching_bi_flat95_fr.tsv`, `submission/fr_claude/candidate_bi_flat95_fr.tsv`
   (`--no-meta`: the optional decision layer in that script is switched off in the final solution).
   Then the exact-key re-add: a pair rejected at 0.95 but kept by the rank rule 0.60 / 0.75 is added only when its
   order-free core-name key AND its (non-empty) order-free address key both equal the Source-1 entity's (no other gate;
   validated leave-one-country-out at corrected 2x: unseen India 495 added / 493 true, unseen US 277 / 276):
   ```
   P -u code/business_entity_resolution/src/final/fr_keyeq_readd.py --fin work/fr_claude/fr_test_final_bi_flat95.parquet --tag bi_flat95_keyeq
   ```
   → `submission/fr_claude/matching_bi_flat95_keyeq_fr.tsv` (863,480 France matches), `candidate_bi_flat95_keyeq_fr.tsv`
9. **Final files** (+ official validator with `--check-ids` if `utils/validate_submission.py` is present):
   ```
   P code/business_entity_resolution/src/final/make_final.py --out output --fr-tag bi_flat95_keyeq
   ```

Steps 7-9 were re-run from this package on 27 Sep and reproduced both output files byte-for-byte (the sha256 above).
Some scripts also read logit caches written by earlier runs when present (`work/audit/...`); when absent, the same fold
models score those pairs. GPU nondeterminism can change a few logits in the last digits.

## Measured (training data only; never test labels)
| system | where | candidates / S1 (test) | macro F0.5 |
|---|---|---|---|
| India + US final (steps 1-7) | out-of-fold, all 2.2M training S1, 1x | 4.623 | 0.991798 |
| India + US final | out-of-fold, corrected 2x (test density) | 4.623 | 0.991349 (India 0.992119, US 0.990836) |
| France chain (step 8), leave-one-country-out, source-only training, 2x, flat 0.95 | unseen US / unseen India | 5.602 | 0.98304 / 0.96808 |
| + exact-key re-add (same 250k-S1 sample: flat 0.95 → with rule) | unseen US / unseen India | 5.602 | 0.98283 → 0.98293 / 0.96746 → 0.96792 |
Public leaderboard: 0.987133 for the same files without the 111 re-added France matches (this exact file: not scored).
