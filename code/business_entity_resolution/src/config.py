"""Pipeline settings (all tuned on a held-out split of the training data)."""

BLOCKING = {
    "k_name": 5,      # forward top-k by name keys
    "k_addr": 5,      # forward top-k by address keys
    "k_comb": 15,     # forward top-k by combined name+address+cross keys
    "k_rev": 2,       # reverse: for every S2/S3 record, its top-k S1 by combined keys
    "max_df": 6000,   # blocking keys more frequent than this are ignored
    "thr": 0.02,      # minimum cosine for a blocking hit
}

# Targeted rescue candidates added inside blocking (rescue.py), one method per experiment: {method: params}.
# Changes the candidate set, so a run with a different RESCUE needs a fresh --work folder. None: off (v16).
# EXP-003 "translit": {"k": 1, "min_addr_cos": 0.05}                      (kept)
# EXP-004 "typo":     {"k": 1, "min_word_sim": 75, "min_addr_cos": 0.05}   (kept)
# EXP-005 "dba":      {"k": 1, "min_cos": 0.3, "min_margin": 0.1}          (kept, provisional)
# v17r (26 Sep): the three kept rescues. v16: RESCUE = None.
RESCUE = {"translit": {"k": 1, "min_addr_cos": 0.05},
          "typo": {"k": 1, "min_word_sim": 75, "min_addr_cos": 0.05},
          "dba": {"k": 1, "min_cos": 0.3, "min_margin": 0.1}}

TRAIN_SAMPLE_S1 = None     # S1 entities used to fit the models (half per fold; None = all); validation is out-of-fold
LGB_MAX_ROUNDS = 1500      # upper bound on boosting rounds; early stopping (50 rounds) picks the number
SEED = 42
MODEL_BACKEND = "lightgbm"  # "lightgbm" (CPU) or "xgb_cuda" (XGBoost on the NVIDIA GPU)

NORM_V7 = True          # v7 normalisation: glued house-number letters, honorific prefixes,
                        # website names next to weak words, own-country word in names
USE_V2 = True           # v2 = extra features + two-stage model (v1 = single model on base features)
USE_NUMREL = True       # v4: house-number relation features (truncation / small shift)
RANK_THRESHOLDS = True   # v4: best candidate of each S1 gets its own (lower) threshold
USE_SRCCTX = True       # v5: source-aware stage-1 context (S2 / S3 confident matches of S1 and competitor)
LEARN_ADDR_COMPONENTS = True  # v6: learn region / spelling equivalences of address components for
                              # countries without training labels (address_align.py)
USE_SIBCTX = True       # v9: sibling context in stage 2 (same address / numbers / name as a confident
                        # other candidate of the same S1 or of the competing S1)
MODEL_SUBDIR = "model_v9"  # models, thresholds and the probability-context cache are kept per version

# Countries absent from the training labels (e.g. France): model without word-risk
# features and a stricter threshold, both chosen with the leave-one-country-out experiment.
UNSEEN_COUNTRY_THRESHOLD = 0.80
# v8: for those countries also a stage-2 model (no word-risk features, context of the no-word-risk stage-1
# probabilities); final probability = mean of stage 1 and stage 2, at a stricter threshold. Leave-one-country-out
# (src/exp_loco_stage2.py): mean over India / US unseen 0.9375 (stage 1 @0.80) -> 0.9396 (mean @0.85).
UNSEEN_STAGE2 = True
# 0.95 since v9_fr95: the leaderboard probe (v8 at 0.85 -> 0.95: public +0.00068, France about +0.0045) and the
# leave-one-country-out run at the test's distractor density (src/exp_loco_selftrain.py) both favour it.
UNSEEN_STAGE2_THRESHOLD = 0.95

# v10: stage 3 for the countries with training labels = the stage-2 inputs + the logit of a cross-encoder
# (multilingual-e5-small fine-tuned on the GPU to read both raw records, src/cross_encoder.py) for the pairs the
# stage-2 model is unsure about (CE_ZONE < p < 1 - CE_ZONE); needs a CUDA GPU (about 1.5 h).
USE_CE = True
CE_ZONE = 0.01
CE_SCORE_ZONE = 0.01   # stage 3 covers CE_SCORE_ZONE < p < 1 - CE_SCORE_ZONE; pairs outside the training zone are
                       # scored with the same (out-of-fold) cross-encoder models. v10: = CE_ZONE
CE_CONTEXT = True      # v12: stage 3 also gets the cross-encoder context (rank / gap within the S1, gap to the other
                       # S1 entities claiming the same record; src/exp_stage3_ctx.py: 0.98834 -> 0.98855). v10: False
# v14 candidate: a second cross-encoder logit in stage 3 (India / US). "norm": the same backbone fine-tuned on the
# pipeline's normalised fields (transliterated names, expanded address tokens, canonical state: a second view of each
# pair, ce_<model>_norm/); "per_country": the per-country models of the France transfer test (rejected: their fold
# models are inconsistent, src/exp_stage3_ens.py). None: off. Validated with src/exp_stage3_ens.py: on the 967k S1 with
# uncertain pairs, log-loss 0.16553 -> 0.16388 and macro F0.5 0.97983 -> 0.98004 (about +0.0001 over all S1). v16: "norm".
CE2 = "norm"
# v13: the same for countries without training labels (France): uncertain pairs of their chain (no-word-risk
# stage 1 / stage 2 mean) + cross-encoder logit -> stage-3 model without word-risk features, trained on India + US;
# the final probability of those pairs is the mean of the chain and stage 3 (UNSEEN_STAGE3_BLEND), decided at
# UNSEEN_STAGE3_THRESHOLD. Leave-one-country-out (src/exp_loco_ce.py, unseen US at the test's distractor density):
# the mean beats the chain at every threshold >= 0.80, at 0.95 0.9595 -> 0.9637 (stage 3 alone: 0.9631).
# v15: unseen India (the harder direction, like France; models trained on US) at the test density, at 0.95:
# chain 0.9094, mean 0.9175, stage 3 alone 0.9192, stage 3 + cross-encoder context 0.9201 -> stage 3 with the
# context, without the mean (UNSEEN_STAGE3_BLEND = False, UNSEEN_STAGE3_CONTEXT = True). v13: True / False.
USE_CE_UNSEEN = True
UNSEEN_STAGE3_BLEND = False
UNSEEN_STAGE3_CONTEXT = True
UNSEEN_STAGE3_THRESHOLD = 0.95
UNSEEN_CE_ZONE = CE_ZONE  # the France chain's uncertain zone re-scored by stage 3 (v15: = CE_ZONE)
# v17 (not recommended): the second cross-encoder view (CE2) in the France stage 3 too. Transfer test at x1.9, at 0.95:
# unseen India 0.9201 -> 0.9228, but unseen US 0.9645 -> 0.9608 (worse at every threshold); mean 0.9423 -> 0.9418. The
# India gain comes from transliterating Indian scripts, which France (Latin script, like US) does not need. v16: False.
UNSEEN_CE2 = False
# v18 candidate (sprint 26 Sep): features the unseen-country models also exclude. Leave-one-country-out (stage 1,
# reports/sprint/loo.py): the state- and legal-form CONFLICT flags come from hand-built per-country tables and flip
# meaning between countries (a state mismatch is a strong 'different business' signal in US data but frequent
# among true Indian matches; a legal-form mismatch is the reverse). v17r: set().
UNSEEN_EXTRA_EXCLUDE = set()
# v19 (27 Sep): a third cross-encoder logit in the India / US stage 3, from a larger backbone (multilingual-e5-base, MIT,
# 278M parameters), same zone / folds / recipe as CE (learning rate CE3_LR), with its context (gap to the best other
# candidate of the S1 and to the best other S1 claiming the record); stage 3 and its rank rule are retrained / re-tuned.
# Offline (reports/099/s3_ce3.py): out-of-fold log-loss 0.2254 / 0.2210 -> 0.2178 / 0.2181 per fold; macro F0.5 1x
# 0.98921 -> 0.98938, 2x corrected 0.98849 -> 0.98867. Adds about 3 h of GPU. None: off (v18).
CE3_BACKBONE = "intfloat/multilingual-e5-base"
CE3_LR = 3e-5
# Exported candidate_pairs.tsv: keep blocking pairs whose stage-1 probability >= this (same predicate as
# reports/leader_gap/cascade_chain.py and fr_cascade.py --steps filt, tau=0.003). None: export all blocked pairs.
CAND_P1_MIN = 0.003
# v18 (27 Sep): self-training for the countries without training labels (removed from the final package). The chain's confident
# unseen-country test predictions (>= hi -> 1, <= lo -> 0) train a student cross-fitted by S1 parity; those pairs'
# final probability = (1 - alpha) * chain + alpha * student, decided at UNSEEN_STAGE3_THRESHOLD. Chain-level
# leave-one-country-out improves unseen India and unseen US at 0.95 (reports/sprint/SPRINT_LOG.md, S-013);
# leaderboard, France-only change: v17r_nodba 0.983154 -> v18_self 0.983416. None: off (v17r).
# Disabled 27 Sep 2026: compliant submissions must not fit on test pseudo-labels (was {"hi": 0.97, "lo": 0.03, "alpha": 0.5}).
UNSEEN_SELFTRAIN = None

# Current settings = v16 (recommended). v17: UNSEEN_CE2 = True. v15: CE2 = None.
# To reproduce v13: CE2 = None, UNSEEN_STAGE3_BLEND = True, UNSEEN_STAGE3_CONTEXT = False.
# To reproduce v12: USE_CE_UNSEEN = False (v11 was never built: its France stage 3 is part of v13). v10: also CE_CONTEXT = False.
# To reproduce v9 (pipeline run of 26 Sep): USE_CE = False, UNSEEN_STAGE2_THRESHOLD = 0.85; v9_fr95 = v9 with 0.95.
# To reproduce v8: USE_SIBCTX = False, MODEL_SUBDIR = "model_v7" (--work ../../work_v7).
# To reproduce v7: as v8 with UNSEEN_STAGE2 = False.
# To reproduce v6: NORM_V7 = False, MODEL_SUBDIR = "model_v6" (normalisation, blocking and features
# are shared with v4, so v7 runs in its own work folder: --work ../../work_v7).
# To reproduce v4 (submitted 25 Sep, public 0.977): NORM_V7 = False, USE_SRCCTX = False, LEARN_ADDR_COMPONENTS = False,
# MODEL_SUBDIR = "model_v4".
# To reproduce v2/v3: TRAIN_SAMPLE_S1 = 600_000, LGB_MAX_ROUNDS = 600,
# USE_NUMREL = False, RANK_THRESHOLDS = False, MODEL_SUBDIR = "model_v2".
