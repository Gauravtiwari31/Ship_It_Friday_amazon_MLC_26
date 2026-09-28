"""End-to-end pipeline: data -> normalisation -> blocking -> features -> model -> outputs.

Usage (from code/business_entity_resolution/):
    python src/run_pipeline.py --data <dataset dir> --work <scratch dir> --out <output dir>

Every stage caches its result in --work, so a re-run resumes where it stopped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_blocking  # noqa: E402
import run_features  # noqa: E402
from decide import decide, tune_rank_thresholds, tune_threshold  # noqa: E402
from io_utils import load_ground_truth, write_id_lists  # noqa: E402
from prepare import build_translit, normalize_split  # noqa: E402
from config import (CE2, CE_CONTEXT, UNSEEN_CE2, UNSEEN_CE_ZONE, UNSEEN_STAGE3_CONTEXT, CE_SCORE_ZONE, CE_ZONE, LEARN_ADDR_COMPONENTS, MODEL_SUBDIR,  # noqa: E402
                    RANK_THRESHOLDS,
                    UNSEEN_COUNTRY_THRESHOLD, UNSEEN_STAGE2, UNSEEN_STAGE2_THRESHOLD, UNSEEN_STAGE3_THRESHOLD,
                    UNSEEN_STAGE3_BLEND, USE_CE, USE_CE_UNSEEN, USE_NUMREL, USE_SIBCTX, USE_SRCCTX, USE_V2)
from address_align import refine_unseen_addresses  # noqa: E402
from features import WORD_RISK_FEATURES  # noqa: E402
from config import CAND_P1_MIN, CE3_BACKBONE, CE3_LR, UNSEEN_EXTRA_EXCLUDE, UNSEEN_SELFTRAIN  # noqa: E402

# features the models of countries without labels never see: the word-risk encodings (learned from labelled
# vocabulary) + config.UNSEEN_EXTRA_EXCLUDE
UNSEEN_EXCLUDE = set(WORD_RISK_FEATURES) | set(UNSEEN_EXTRA_EXCLUDE)
from features_extra import build_extra  # noqa: E402
from features_num import build_numrel  # noqa: E402
from stage2 import (ce_context, write_context_parts, write_sibling_context_parts,  # noqa: E402
                    write_source_context_parts)
from train import load_part, part_files, score_parts, train_all  # noqa: E402

T0 = time.time()
LOG_FILE = None


def log(msg):
    """Print a time-stamped progress line and append it to <work>/pipeline_log.txt."""
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def tuned_threshold(path, oof, gt, train_s1, label):
    """Load the tuned threshold from `path`, or tune it on out-of-fold scores and save it."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["threshold"]
    (thr, score), _ = tune_threshold(oof, gt, train_s1, log=log)
    log(f"  {label}: best threshold {thr:.2f}, out-of-fold macro F0.5 = {score:.5f}")
    with open(path, "w") as f:
        json.dump({"threshold": thr, "oof_macro_f05": score}, f)
    return thr


def run_v1(work, feat_train, feat_test, gt, train_s1):
    """v1: one LightGBM on the base features."""
    model_dir = os.path.join(work, "model")
    log("stage 4: model training (2-fold cross-fitting)")
    models = train_all([feat_train], model_dir, prefix="lgb", log=log)
    log("stage 5: threshold tuning on out-of-fold training scores")
    thr_path = os.path.join(model_dir, "threshold.json")
    if not os.path.exists(thr_path):
        tuned_threshold(thr_path, score_parts([feat_train], models, oof=True, log=log), gt, train_s1, "v1")
    log("stage 6: test predictions")
    return score_parts([feat_test], models, oof=False, log=log)


def _part_sizes(d):
    """Row count of every part file of a feature directory."""
    return [pl.scan_parquet(f).select(pl.len()).collect().item() for f in part_files(d)]


def stage2_thresholds(path, oof, gt, train_s1):
    """Tune (or load) the stage-2 decision thresholds: one flat threshold and, if
    RANK_THRESHOLDS, the rank rule (best candidate of each S1 / its other candidates).
    Returns (thr_top, thr_rest)."""
    if not os.path.exists(path):
        (thr, score), _ = tune_threshold(oof, gt, train_s1, log=log)
        log(f"  stage-2: best flat threshold {thr:.2f}, out-of-fold macro F0.5 = {score:.5f}")
        res = {"threshold": thr, "oof_macro_f05": score}
        if RANK_THRESHOLDS:
            t_top, t_rest, s_rank = tune_rank_thresholds(oof, gt, train_s1, log=log)
            log(f"  stage-2: rank rule top={t_top:.2f} rest={t_rest:.2f}, out-of-fold macro F0.5 = {s_rank:.5f}")
            res.update(threshold_top=t_top, threshold_rest=t_rest, oof_macro_f05_rank=s_rank)
        with open(path, "w") as f:
            json.dump(res, f)
    with open(path) as f:
        res = json.load(f)
    if RANK_THRESHOLDS and "threshold_top" in res:
        return res["threshold_top"], res["threshold_rest"]
    return res["threshold"], res["threshold"]


def run_v2(work, feat_train, feat_test, gt, train_s1):
    """v2 / v4: base + extra (+ house-number relation) features -> stage-1 LightGBM ->
    probability context -> stage-2 LightGBM -> thresholds tuned out-of-fold."""
    model_dir = os.path.join(work, MODEL_SUBDIR)
    os.makedirs(model_dir, exist_ok=True)
    log("stage 3b: extra features (name/address rarity, word-risk encoding)")
    dirs = {}
    for split, fdir in (("train", feat_train), ("test", feat_test)):
        edir = os.path.join(work, f"{split}_extra")
        if len(part_files(edir)) != len(part_files(fdir)):
            build_extra(work, split, part_files(fdir), log=log)
        dirs[split] = [fdir, edir]
    if USE_NUMREL:
        log("stage 3c: house-number relation features (truncation / shift)")
        for split, fdir in (("train", feat_train), ("test", feat_test)):
            ndir = os.path.join(work, f"{split}_numrel")
            if len(part_files(ndir)) != len(part_files(fdir)):
                build_numrel(work, split, part_files(fdir), log=log)
            dirs[split].append(ndir)
    base = {split: list(d) for split, d in dirs.items()}  # stage-1 inputs (no probability context)

    log("stage 4: stage-1 model (2-fold cross-fitting)")
    m1 = train_all(dirs["train"], model_dir, prefix="stage1", log=log)

    log("stage 4b: stage-1 probabilities -> context features" + (" (+ source-aware)" if USE_SRCCTX else "")
        + (" (+ sibling)" if USE_SIBCTX else ""))
    for split in ("train", "test"):
        cdir = os.path.join(work, f"{split}_p1ctx_{MODEL_SUBDIR}")
        sdir = os.path.join(work, f"{split}_srcctx_{MODEL_SUBDIR}")
        zdir = os.path.join(work, f"{split}_sibctx_{MODEL_SUBDIR}")
        n = len(part_files(dirs[split][0]))
        need_c = len(part_files(cdir)) != n
        need_s = USE_SRCCTX and len(part_files(sdir)) != n
        need_z = USE_SIBCTX and len(part_files(zdir)) != n
        if need_c or need_s or need_z:
            sc = score_parts(dirs[split], m1, oof=(split == "train"), log=log)
            if split == "train":
                tuned_threshold(os.path.join(model_dir, "threshold_stage1.json"), sc, gt, train_s1, "stage-1")
            sizes = _part_sizes(dirs[split][0])
            if need_c:
                write_context_parts(sc, sizes, cdir)
            if need_s:
                cand_src = pl.read_parquet(os.path.join(work, f"{split}_norm.parquet"), columns=["entity_id", "src"])
                write_source_context_parts(sc, cand_src, sizes, sdir)
            if need_z:
                write_sibling_context_parts(sc, _cand_attr(work, split), sizes, zdir)
            del sc
        dirs[split] = dirs[split] + [cdir] + ([sdir] if USE_SRCCTX else []) + ([zdir] if USE_SIBCTX else [])

    log("stage 4c: stage-2 model (2-fold cross-fitting)")
    m2 = train_all(dirs["train"], model_dir, prefix="stage2", log=log)

    log("stage 5: threshold tuning on out-of-fold stage-2 scores")
    thr_path = os.path.join(model_dir, "threshold.json")
    oof = None if os.path.exists(thr_path) else score_parts(dirs["train"], m2, oof=True, log=log)
    if oof is not None:  # kept for threshold / error analyses (src/exp_test_shift.py, src/analysis_v2.py)
        oof.write_parquet(os.path.join(work, f"train_oof_{MODEL_SUBDIR}.parquet"))
    thr_top, thr_rest = stage2_thresholds(thr_path, oof, gt, train_s1)
    del oof
    log(f"stage 6: test predictions (threshold best candidate {thr_top:.2f}, others {thr_rest:.2f})")
    scored = score_parts(dirs["test"], m2, oof=False, log=log) \
        .with_columns(pl.lit(thr_top).alias("thr_top"), pl.lit(thr_rest).alias("thr_rest"))

    unseen_ids = _unseen_country_s1(work)
    if unseen_ids:
        log(f"stage 6b: {len(unseen_ids):,} test S1 in countries without training labels -> model without "
            f"word-risk features, threshold {UNSEEN_COUNTRY_THRESHOLD:.2f} (chosen by leave-one-country-out)")
        mu = train_all(base["train"], model_dir, prefix="unseen", log=log, exclude=UNSEEN_EXCLUDE)
        su = score_parts(base["test"], mu, oof=False, log=log, s1_ids=unseen_ids)
        thr_u = UNSEEN_COUNTRY_THRESHOLD
        if UNSEEN_STAGE2:
            su, thr_u = unseen_stage2(work, base, mu, su, unseen_ids), UNSEEN_STAGE2_THRESHOLD
        su = su.with_columns(pl.lit(thr_u).alias("thr_top"), pl.lit(thr_u).alias("thr_rest"))
        scored = pl.concat([scored.filter(~pl.col("s1_id").is_in(pl.Series(unseen_ids).implode())),
                            su.select(scored.columns)])
    return scored, dirs


def write_zone_parts(dirs, ce, out_dir, context=False, ce2=None):
    """The uncertain pairs only (those with a cross-encoder logit): all stage-2 inputs + `p2_ce_logit`
    (+ the cross-encoder context, + the per-country second logit `p2_ce2_logit`), in part order
    (a few % of the rows, so stage 3 trains in minutes)."""
    os.makedirs(out_dir, exist_ok=True)
    extra = [ce_context(ce)] if context else []
    ce = pl.concat([ce.select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce_logit"))] + extra,
                   how="horizontal")
    if ce2 is not None:
        ce = ce.join(ce2, on=["s1_id", "cand_id"], how="left", maintain_order="left")
    for i in range(len(part_files(dirs[0]))):
        load_part(i, dirs).join(ce, on=["s1_id", "cand_id"], how="inner", maintain_order="left") \
            .write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))


def _stage3_tag():
    """Name suffix of the stage-3 artefacts (v10: none), so other settings never reuse v10's models."""
    return (f"_z{CE_SCORE_ZONE:g}" if CE_SCORE_ZONE != CE_ZONE else "") + ("_ctx" if CE_CONTEXT else "") \
        + (f"_ce2{CE2}" if CE2 else "")


def _second_scores(data, work):
    """v14 candidate: a second cross-encoder logit for stage 3, {split: (s1_id, cand_id, p2_ce2_logit)} (config.CE2)."""
    if CE2 == "per_country":
        return _per_country_scores(data, work)
    import cross_encoder
    d = os.path.join(work, f"ce_{MODEL_SUBDIR}_{CE2}")
    if not all(os.path.exists(os.path.join(d, f"{s}_scores.parquet")) for s in ("train", "test")):
        log(f"stage 7a: second cross-encoder, view '{CE2}' (GPU, about 1.5 h)")
        cross_encoder.run(data, work, MODEL_SUBDIR, d, zone=CE_ZONE, logger=log, view=CE2)
    return {s: pl.read_parquet(os.path.join(d, f"{s}_scores.parquet"))
            .select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce2_logit")) for s in ("train", "test")}


def _per_country_scores(data, work):
    """v14 candidate: second cross-encoder logits from per-country models (ce_<model>_<country>/, trained on that
    country's uncertain pairs): train pairs out-of-fold, test pairs of the country by the mean of its two
    models. Returns {split: (s1_id, cand_id, p2_ce2_logit)}."""
    import cross_encoder
    s1 = {s: pl.read_parquet(os.path.join(work, f"{s}_norm.parquet"), columns=["entity_id", "src", "country"])
          .filter(pl.col("src") == 1) for s in ("train", "test")}
    out = {"train": [], "test": []}
    for c in sorted(s1["train"]["country"].unique().to_list()):
        d = os.path.join(work, f"ce_{MODEL_SUBDIR}_{c}")
        if not os.path.exists(os.path.join(d, "train_scores.parquet")):
            log(f"stage 7a: per-country cross-encoder for {c} (GPU)")
            cross_encoder.run(data, work, MODEL_SUBDIR, d, zone=CE_ZONE, country=c, no_test=True, logger=log)
        tp = os.path.join(d, "test_scores.parquet")
        if not os.path.exists(tp) or pl.read_parquet(tp).height == 0:
            ids = s1["test"].filter(pl.col("country") == c)["entity_id"]
            z = pl.read_parquet(os.path.join(work, f"test_scored_{MODEL_SUBDIR}.parquet"), columns=["s1_id", "cand_id", "prob"]) \
                .filter((pl.col("prob") > CE_ZONE) & (pl.col("prob") < 1 - CE_ZONE) & pl.col("s1_id").is_in(ids.implode()))
            log(f"  {c}: scoring {z.height:,} uncertain test pairs with its per-country cross-encoder")
            cross_encoder.score_pairs(data, "test", d, z.select("s1_id", "cand_id"), logger=log).write_parquet(tp)
        for split in ("train", "test"):
            out[split].append(pl.read_parquet(os.path.join(d, f"{split}_scores.parquet")))
    return {s: pl.concat(v).select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce2_logit"))
            for s, v in out.items()}


def _wide_scores(data, work, split, ce_dir, have, unseen):
    """Cross-encoder logits for the pairs of the wider stage-3 zone that are outside the training zone
    (train: out-of-fold fold models; test: countries with labels, mean of both models); cached."""
    import cross_encoder
    path = os.path.join(ce_dir, f"{split}_scores_wide{CE_SCORE_ZONE:g}.parquet")
    if not os.path.exists(path):
        z = (pl.col("prob") > CE_SCORE_ZONE) & (pl.col("prob") < 1 - CE_SCORE_ZONE)
        if split == "train":
            folds = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"])
                               for f in part_files(os.path.join(work, "train_feats"))]).unique("s1_id")
            src = pl.read_parquet(os.path.join(work, f"train_oof_{MODEL_SUBDIR}.parquet"), columns=["s1_id", "cand_id", "prob"]) \
                .filter(z).join(folds, on="s1_id")
        else:
            src = pl.read_parquet(os.path.join(work, f"test_scored_{MODEL_SUBDIR}.parquet"), columns=["s1_id", "cand_id", "prob"]) \
                .filter(z & ~pl.col("s1_id").is_in(pl.Series(list(unseen)).implode()))
        todo = src.join(have, on=["s1_id", "cand_id"], how="anti")
        log(f"  {split}: {todo.height:,} pairs of the wider zone {CE_SCORE_ZONE:g} < p < {1 - CE_SCORE_ZONE:g} to score")
        cross_encoder.score_pairs(data, split, ce_dir, todo, logger=log).write_parquet(path)
    return pl.read_parquet(path)


def _third_zone_parts(data, work, split, zone_dir):
    """v19: the third cross-encoder (config.CE3_BACKBONE, raw records) for the stage-3 zone: parts row-aligned with the
    zone parts holding p2_ce3_logit and its context (gap to the best other candidate of the S1 and to the best other S1
    claiming the record). Cached in <work>/<split>_zone_ce3_<model>/; the logits in <work>/ce_<model>_e5base/."""
    import cross_encoder
    out = os.path.join(work, f"{split}_zone_ce3_{MODEL_SUBDIR}")
    if len(part_files(out)) == len(part_files(zone_dir)):
        return out
    d = os.path.join(work, f"ce_{MODEL_SUBDIR}_e5base")
    if not all(os.path.exists(os.path.join(d, f"{s}_scores.parquet")) for s in ("train", "test")):
        log(f"stage 7e: third cross-encoder, backbone {CE3_BACKBONE} (GPU, about 3 h)")
        cross_encoder.run(data, work, MODEL_SUBDIR, d, zone=CE_ZONE, backbone=CE3_BACKBONE, lr=CE3_LR, logger=log)
    ce = pl.read_parquet(os.path.join(d, f"{split}_scores.parquet")).unique(["s1_id", "cand_id"])         .with_columns(pl.col("ce_logit").cast(pl.Float32))
    keys = pl.concat([pl.read_parquet(f, columns=["s1_id", "cand_id"] + (["fold"] if split == "train" else []))
                      for f in part_files(zone_dir)])
    miss = keys.join(ce, on=["s1_id", "cand_id"], how="anti")
    if miss.height:  # zone pairs without a logit (e.g. a wider CE_SCORE_ZONE): score them with the same fold models
        log(f"  third cross-encoder: scoring {miss.height:,} {split} zone pairs without a logit")
        cross_encoder.BACKBONE = CE3_BACKBONE
        ce = pl.concat([ce, cross_encoder.score_pairs(data, split, d, miss, logger=log).with_columns(pl.col("ce_logit").cast(pl.Float32))])
    ctx = ce_context(ce)
    t = pl.concat([ce.select("s1_id", "cand_id", pl.col("ce_logit").alias("p2_ce3_logit")),
                   ctx.select(pl.col("p2_ce_gap_s1").alias("p2_ce3_gap_s1"), pl.col("p2_ce_gap_cand").alias("p2_ce3_gap_cand"))],
                  how="horizontal")
    os.makedirs(out, exist_ok=True)
    for i, f in enumerate(part_files(zone_dir)):
        pl.read_parquet(f, columns=["s1_id", "cand_id"]).join(t, on=["s1_id", "cand_id"], how="left", maintain_order="left")             .drop("s1_id", "cand_id").write_parquet(os.path.join(out, f"part_{i:03d}.parquet"))
    return out


def run_stage3(data, work, dirs, scored, gt, train_s1):
    """v10: stage 3 for the countries with training labels. A cross-encoder (a multilingual transformer
    reading both raw records, src/cross_encoder.py) scores the pairs the stage-2 model is unsure about
    (CE_ZONE < p < 1 - CE_ZONE); a LightGBM on those pairs (stage-2 inputs + the cross-encoder logit),
    cross-fitted like stages 1 and 2, replaces their probability. The other pairs keep the stage-2
    probability; the rank-rule thresholds are re-tuned on the combined out-of-fold scores.
    Countries without training labels keep their stage-2 chain."""
    import cross_encoder
    model_dir = os.path.join(work, MODEL_SUBDIR)
    ce_dir = os.path.join(work, f"ce_{MODEL_SUBDIR}")
    if not all(os.path.exists(os.path.join(ce_dir, f"{s}_scores.parquet")) for s in ("train", "test")):
        log("stage 7: cross-encoder on the uncertain pairs (GPU, about 1.5 h)")
        cross_encoder.run(data, work, MODEL_SUBDIR, ce_dir, zone=CE_ZONE, logger=log)
    tag = _stage3_tag()
    unseen = set(_unseen_country_s1(work))
    log(f"stage 7b: stage-3 model on the uncertain pairs = stage-2 inputs + cross-encoder logit"
        f"{' + its context' if CE_CONTEXT else ''} (zone {CE_SCORE_ZONE:g}; 2-fold cross-fitting)")
    zdir = {}
    ce2 = None
    for split in ("train", "test"):
        zdir[split] = os.path.join(work, f"{split}_zone{tag}_{MODEL_SUBDIR}")
        if len(part_files(zdir[split])) != len(part_files(dirs[split][0])):
            s = pl.read_parquet(os.path.join(ce_dir, f"{split}_scores.parquet"))
            if CE_SCORE_ZONE < CE_ZONE:
                s = pl.concat([s, _wide_scores(data, work, split, ce_dir, s, unseen)])
            if CE2 and ce2 is None:
                ce2 = _second_scores(data, work)
            write_zone_parts(dirs[split], s, zdir[split], context=CE_CONTEXT, ce2=ce2[split] if ce2 else None)
    zdirs = {s: [zdir[s]] for s in ("train", "test")}
    if CE3_BACKBONE:  # v19: + the third (larger) cross-encoder's logit and context, own stage-3 model and thresholds
        tag = tag + "_ce3"
        for split in ("train", "test"):
            zdirs[split].append(_third_zone_parts(data, work, split, zdir[split]))
    m3 = train_all(zdirs["train"], model_dir, prefix=f"stage3{tag}", log=log)
    thr_path = os.path.join(model_dir, f"threshold_stage3{tag}.json")
    oof = None
    if not os.path.exists(thr_path):
        z = score_parts(zdirs["train"], m3, oof=True, log=log).select("s1_id", "cand_id", pl.col("prob").alias("p3"))
        oof = pl.read_parquet(os.path.join(work, f"train_oof_{MODEL_SUBDIR}.parquet"), columns=["s1_id", "cand_id", "prob"]) \
            .join(z, on=["s1_id", "cand_id"], how="left", maintain_order="left") \
            .with_columns(pl.coalesce("p3", "prob").alias("prob")).drop("p3")
        oof.write_parquet(os.path.join(work, f"train_oof_{MODEL_SUBDIR}_stage3{tag}.parquet"))
    thr_top, thr_rest = stage2_thresholds(thr_path, oof, gt, train_s1)
    del oof
    seen_ids = [s for s in scored["s1_id"].unique().to_list() if s not in unseen]
    log(f"stage 7c: stage-3 test predictions for the uncertain pairs of {len(seen_ids):,} S1 of the countries "
        f"with labels (threshold best candidate {thr_top:.2f}, others {thr_rest:.2f})")
    z = score_parts(zdirs["test"], m3, oof=False, log=log, s1_ids=seen_ids).select("s1_id", "cand_id", pl.col("prob").alias("p3"))
    is_seen = ~pl.col("s1_id").is_in(pl.Series(list(unseen)).implode())
    scored = scored.join(z, on=["s1_id", "cand_id"], how="left", maintain_order="left").with_columns(
        pl.coalesce("p3", "prob").alias("prob"),
        pl.when(is_seen).then(pl.lit(thr_top)).otherwise(pl.col("thr_top")).alias("thr_top"),
        pl.when(is_seen).then(pl.lit(thr_rest)).otherwise(pl.col("thr_rest")).alias("thr_rest")).drop("p3")
    if USE_CE_UNSEEN and unseen:
        scored = unseen_stage3(data, work, dirs, scored, sorted(unseen), ce_dir)
    return scored


def unseen_stage3(data, work, dirs, scored, unseen_ids, ce_dir):
    """Stage 3 for countries without training labels (France), as validated by src/exp_loco_ce.py:
    the pairs their chain (mean of the no-word-risk stage 1 and stage 2) is unsure about get a
    cross-encoder logit (train: out-of-fold, from the fold models; test: mean of both models), and a
    no-word-risk model on those pairs (unseen stage-2 inputs + chain probability + logit), trained on
    India + US, replaces their probability; decided at UNSEEN_STAGE3_THRESHOLD."""
    import cross_encoder
    model_dir = os.path.join(work, MODEL_SUBDIR)
    log("stage 7d: stage 3 for countries without training labels (cross-encoder on their uncertain pairs)")
    base = {s: [d for d in dirs[s] if not any(t in os.path.basename(d) for t in ("_p1ctx_", "_srcctx_", "_sibctx_"))]
            for s in ("train", "test")}
    udirs = {s: base[s] + [os.path.join(work, f"{s}_{c}_unseen_{MODEL_SUBDIR}") for c in ("p1ctx", "srcctx")]
             + ([os.path.join(work, f"{s}_sibctx_unseen_{MODEL_SUBDIR}")] if USE_SIBCTX else []) for s in ("train", "test")}
    uz = UNSEEN_CE_ZONE
    zsuf = "" if uz == CE_ZONE else f"_z{uz:g}"
    oof_path = os.path.join(work, f"train_oof_unseen{zsuf}_{MODEL_SUBDIR}.parquet")
    if not os.path.exists(oof_path):  # the chain's out-of-fold probability on India + US (its uncertain pairs only)
        # part by part (memory-lean); the stage-1 out-of-fold probability is already stored as p1_prob (stage 6c)
        mu2 = train_all(udirs["train"], model_dir, prefix="unseen_stage2", log=log, exclude=UNSEEN_EXCLUDE)
        feats, keep = mu2[0].feature_name(), []
        for i in range(len(part_files(udirs["train"][0]))):
            df = load_part(i, udirs["train"])
            X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
            q0, q1 = (m.predict(X, num_threads=os.cpu_count()) for m in mu2)
            p2 = np.where(df["fold"].to_numpy() == 0, q1, q0)  # a pair of fold k is scored by the model of fold 1 - k
            chain = (df["p1_prob"].to_numpy() + p2) / 2
            keep.append(df.select("s1_id", "cand_id", "fold").with_columns(pl.Series("prob", chain.astype(np.float32)))
                        .filter((pl.col("prob") > uz) & (pl.col("prob") < 1 - uz)))
            del df, X
        pl.concat(keep).write_parquet(oof_path)
        log(f"  chain out-of-fold on the training countries: {sum(k.height for k in keep):,} uncertain pairs")
        del keep
    z = pl.read_parquet(oof_path).filter((pl.col("prob") > uz) & (pl.col("prob") < 1 - uz))
    ce_path = os.path.join(ce_dir, f"train_scores_unseen{zsuf}.parquet")
    if not os.path.exists(ce_path):  # logits for the chain's uncertain train pairs (reuse, score the rest)
        have = pl.concat([pl.read_parquet(f, columns=["s1_id", "cand_id", "ce_logit"]) for f in
                          (os.path.join(ce_dir, "train_scores.parquet"), os.path.join(ce_dir, "train_scores_unseen.parquet"))
                          if os.path.exists(f)]).unique(["s1_id", "cand_id"])
        todo = z.join(have, on=["s1_id", "cand_id"], how="anti")
        log(f"  {z.height:,} uncertain train pairs; {z.height - todo.height:,} already scored, scoring {todo.height:,}")
        new = cross_encoder.score_pairs(data, "train", ce_dir, todo, logger=log)
        pl.concat([z.join(have, on=["s1_id", "cand_id"], how="inner").select("s1_id", "cand_id", "ce_logit"), new]) \
            .write_parquet(ce_path)
    tz = scored.filter(pl.col("s1_id").is_in(pl.Series(unseen_ids).implode())
                       & (pl.col("prob") > uz) & (pl.col("prob") < 1 - uz)).select("s1_id", "cand_id", "prob")
    tce = pl.read_parquet(os.path.join(ce_dir, "test_scores.parquet"))
    missing = tz.join(tce, on=["s1_id", "cand_id"], how="anti")
    if missing.height:
        log(f"  scoring {missing.height:,} uncertain test pairs without a logit")
        tce = pl.concat([tce, cross_encoder.score_pairs(data, "test", ce_dir, missing.select("s1_id", "cand_id"), logger=log)])
    tabs = {"train": z.select("s1_id", "cand_id", pl.col("prob").alias("p2_unseen_base"))
            .join(pl.read_parquet(ce_path), on=["s1_id", "cand_id"]),
            "test": tz.select("s1_id", "cand_id", pl.col("prob").alias("p2_unseen_base"))
            .join(tce, on=["s1_id", "cand_id"])}
    if UNSEEN_CE2 and CE2:  # v17 candidate: the second cross-encoder view (CE2) in this stage 3 as well
        d2 = os.path.join(work, f"ce_{MODEL_SUBDIR}_{CE2}")
        p2 = os.path.join(d2, f"train_scores_unseen{zsuf}.parquet")
        if not os.path.exists(p2):
            have2 = pl.read_parquet(os.path.join(d2, "train_scores.parquet"))
            todo2 = z.join(have2, on=["s1_id", "cand_id"], how="anti")
            log(f"  second view: scoring {todo2.height:,} uncertain train pairs")
            new2 = cross_encoder.score_pairs(data, "train", d2, todo2, logger=log, view=CE2, work=work)
            pl.concat([z.join(have2, on=["s1_id", "cand_id"], how="inner").select("s1_id", "cand_id", "ce_logit"), new2])                 .write_parquet(p2)
        tce2 = pl.read_parquet(os.path.join(d2, "test_scores.parquet"))
        miss2 = tz.join(tce2, on=["s1_id", "cand_id"], how="anti")
        if miss2.height:
            log(f"  second view: scoring {miss2.height:,} uncertain test pairs without a logit")
            tce2 = pl.concat([tce2, cross_encoder.score_pairs(data, "test", d2, miss2.select("s1_id", "cand_id"), logger=log,
                                                              view=CE2, work=work)])
        for split, c2 in (("train", pl.read_parquet(p2)), ("test", tce2)):
            tabs[split] = tabs[split].join(c2.select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce2_logit")),
                                           on=["s1_id", "cand_id"], how="left", maintain_order="left")
    utag = zsuf + ("_ctx" if UNSEEN_STAGE3_CONTEXT else "") + (f"_ce2{CE2}" if UNSEEN_CE2 and CE2 else "")
    zdir = {}
    for split in ("train", "test"):
        zdir[split] = os.path.join(work, f"{split}_uzone{utag}_{MODEL_SUBDIR}")
        if len(part_files(zdir[split])) != len(part_files(udirs[split][0])):
            t = tabs[split].with_columns(pl.col("p2_unseen_base").cast(pl.Float32), pl.col("ce_logit").cast(pl.Float32))
            if UNSEEN_STAGE3_CONTEXT:  # the cross-encoder context over the split's uncertain pairs, as in v12
                t = pl.concat([t, ce_context(t.select("s1_id", "cand_id", "ce_logit"))], how="horizontal")
            os.makedirs(zdir[split], exist_ok=True)
            for i in range(len(part_files(udirs[split][0]))):
                load_part(i, udirs[split]).join(t.rename({"ce_logit": "p2_ce_logit"}), on=["s1_id", "cand_id"],
                                                how="inner", maintain_order="left") \
                    .write_parquet(os.path.join(zdir[split], f"part_{i:03d}.parquet"))
    mu3 = train_all([zdir["train"]], model_dir, prefix=f"unseen_stage3{utag}", log=log, exclude=UNSEEN_EXCLUDE)
    p3 = score_parts([zdir["test"]], mu3, oof=False, log=log).select("s1_id", "cand_id", pl.col("prob").alias("p3"))
    if UNSEEN_STAGE3_BLEND:  # mean with the chain probability: stage 3 alone is over-confident on an unseen country
        p3 = p3.join(tz.select("s1_id", "cand_id", pl.col("prob").alias("_base")), on=["s1_id", "cand_id"],
                     how="left", maintain_order="left") \
            .with_columns(((pl.col("p3") + pl.col("_base")) / 2).alias("p3")).drop("_base")
    log(f"  {p3.height:,} uncertain pairs of {len(unseen_ids):,} S1 rescored{' (mean with the chain)' if UNSEEN_STAGE3_BLEND else ''}; "
        f"threshold {UNSEEN_STAGE3_THRESHOLD:.2f}")
    is_u = pl.col("s1_id").is_in(pl.Series(unseen_ids).implode())
    return scored.join(p3, on=["s1_id", "cand_id"], how="left", maintain_order="left").with_columns(
        pl.coalesce("p3", "prob").alias("prob"),
        pl.when(is_u).then(pl.lit(UNSEEN_STAGE3_THRESHOLD)).otherwise(pl.col("thr_top")).alias("thr_top"),
        pl.when(is_u).then(pl.lit(UNSEEN_STAGE3_THRESHOLD)).otherwise(pl.col("thr_rest")).alias("thr_rest")).drop("p3")


def unseen_stage2(work, base, mu, su, unseen_ids):
    """Stage 2 for countries without training labels (v8): the probabilities of the no-word-risk
    stage-1 model `mu` (out-of-fold on train) are put in context (p1_* and source-aware), a
    no-word-risk stage-2 model is cross-fitted on them, and the returned probability of the
    unseen-country pairs `su` is the mean of stage 1 and stage 2."""
    log("stage 6c: stage 2 for countries without training labels (mean with stage 1)")
    model_dir = os.path.join(work, MODEL_SUBDIR)
    udirs = {}
    for split in ("train", "test"):
        cdir = os.path.join(work, f"{split}_p1ctx_unseen_{MODEL_SUBDIR}")
        sdir = os.path.join(work, f"{split}_srcctx_unseen_{MODEL_SUBDIR}")
        zdir = os.path.join(work, f"{split}_sibctx_unseen_{MODEL_SUBDIR}")
        n = len(part_files(base[split][0]))
        need = [len(part_files(d)) != n for d in (cdir, sdir)] + [USE_SIBCTX and len(part_files(zdir)) != n]
        if any(need):
            sc = score_parts(base[split], mu, oof=(split == "train"), log=log)
            sizes = _part_sizes(base[split][0])
            if need[0]:
                write_context_parts(sc, sizes, cdir)
            if need[1]:
                cand_src = pl.read_parquet(os.path.join(work, f"{split}_norm.parquet"), columns=["entity_id", "src"])
                write_source_context_parts(sc, cand_src, sizes, sdir)
            if need[2]:
                write_sibling_context_parts(sc, _cand_attr(work, split), sizes, zdir)
            del sc
        udirs[split] = base[split] + [cdir, sdir] + ([zdir] if USE_SIBCTX else [])
    mu2 = train_all(udirs["train"], model_dir, prefix="unseen_stage2", log=log, exclude=UNSEEN_EXCLUDE)
    su2 = score_parts(udirs["test"], mu2, oof=False, log=log, s1_ids=unseen_ids)
    assert su2["s1_id"].equals(su["s1_id"]) and su2["cand_id"].equals(su["cand_id"])  # same parts, same filter
    return su.with_columns(((pl.col("prob") + su2["prob"]) / 2).alias("prob"))


def _cand_attr(work, split):
    """Address tokens, house numbers and core name of every record of a split (sibling context)."""
    return pl.read_parquet(os.path.join(work, f"{split}_norm.parquet"),
                           columns=["entity_id", "addr_tokens", "addr_nums", "name_core"])


def _unseen_country_s1(work):
    """Test S1 ids whose country label does not occur among the training S1 records."""
    def s1(split):
        """Country labels of the S1 records of a split."""
        return pl.read_parquet(os.path.join(work, f"{split}_norm.parquet"), columns=["entity_id", "src", "country"]) \
            .filter(pl.col("src") == 1)
    seen = set(s1("train")["country"].unique().to_list())
    return s1("test").filter(~pl.col("country").is_in(list(seen)))["entity_id"].to_list()


def _test_feat_dirs(work):
    """Row-aligned test feature part directories (same layout as cascade_chain.py BASE['test'])."""
    dirs = [os.path.join(work, "test_feats"), os.path.join(work, "test_extra")]
    if USE_NUMREL:
        dirs.append(os.path.join(work, "test_numrel"))
    return dirs


def export_test_candidates(work, tau):
    """Blocking pairs kept when stage-1 p1 >= tau (cascade_chain.py / fr_cascade.py filt on test).

    Countries with training labels use `<work>/test_p1ctx_<MODEL_SUBDIR>`; countries without labels use
    `<work>/test_p1ctx_unseen_<MODEL_SUBDIR>`. Matcher scores are unchanged; matches must be restricted to this set."""
    unseen = set(_unseen_country_s1(work))
    p1_seen = os.path.join(work, f"test_p1ctx_{MODEL_SUBDIR}")
    p1_unseen = os.path.join(work, f"test_p1ctx_unseen_{MODEL_SUBDIR}")
    base = _test_feat_dirs(work)
    kept = []
    n0 = 0
    for i in range(len(part_files(base[0]))):
        df = load_part(i, base).select("s1_id", "cand_id")
        p1s = pl.read_parquet(part_files(p1_seen)[i], columns=["p1_prob"])["p1_prob"]
        p1u = pl.read_parquet(part_files(p1_unseen)[i], columns=["p1_prob"])["p1_prob"]
        assert df.height == p1s.len() == p1u.len()
        is_u = df["s1_id"].is_in(pl.Series(list(unseen)).implode())
        p1 = pl.when(is_u).then(p1u).otherwise(p1s)
        keep = p1 >= tau
        n0 += df.height
        kept.append(df.filter(keep))
    out = pl.concat(kept)
    log(f"candidate export: stage-1 p1 >= {tau:g} kept {out.height:,} of {n0:,} blocked test pairs")
    return out


def main():
    """Run all stages and write candidate_pairs.tsv and matching_results.tsv."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset directory containing train/ and test/")
    ap.add_argument("--work", required=True, help="scratch directory for intermediate files")
    ap.add_argument("--out", required=True, help="output directory for the two TSV files")
    args = ap.parse_args()
    model_dir = os.path.join(args.work, "model")
    os.makedirs(model_dir, exist_ok=True)
    global LOG_FILE
    LOG_FILE = os.path.join(args.work, "pipeline_log.txt")

    log("stage 1: normalisation")
    tmap = build_translit(args.data, args.work, model_dir)
    for split in ("train", "test"):
        normalize_split(args.data, split, args.work, tmap)
    load_ground_truth(args.data, args.work)

    log("stage 2: blocking")
    for split in ("train", "test"):
        c = run_blocking.run(args.work, split, log=log)
        log(f"  {split}: {c.height:,} candidate pairs")
    gt = pl.read_parquet(os.path.join(args.work, "train_gt_pairs.parquet"))
    if LEARN_ADDR_COMPONENTS:
        log("stage 2b: address-component equivalences for countries without training labels")
        refine_unseen_addresses(args.work, tmap, log=log)

    log("stage 3: pair features")
    feat_train = run_features.run(args.work, "train", log=log)
    feat_test = run_features.run(args.work, "test", log=log)

    train_s1 = pl.read_parquet(os.path.join(args.work, "train_norm.parquet"), columns=["entity_id", "src"]) \
        .filter(pl.col("src") == 1)["entity_id"].to_list()

    if USE_V2:
        scored, s2dirs = run_v2(args.work, feat_train, feat_test, gt, train_s1)
        scored.write_parquet(os.path.join(args.work, f"test_scored_{MODEL_SUBDIR}.parquet"))  # for analysis / stage 7
        if USE_CE:
            scored = run_stage3(args.data, args.work, s2dirs, scored, gt, train_s1)
        # no self-training / pseudo-labels on test (the final solution never fits on test predictions)
        assert not UNSEEN_SELFTRAIN, "test self-training is not part of the final (compliant) solution"
        unseen = []
        if USE_CE or unseen:
            # final probabilities and thresholds, so other decision thresholds can be tried without a re-run
            scored.write_parquet(os.path.join(args.work, f"test_scored_{MODEL_SUBDIR}_final.parquet"))
    else:
        scored = run_v1(args.work, feat_train, feat_test, gt, train_s1)
    if "thr" in scored.columns or "thr_top" in scored.columns:
        matches = decide(scored)  # per-pair (country-dependent) thresholds
    else:
        with open(os.path.join(args.work, "model", "threshold.json")) as f:
            matches = decide(scored, json.load(f)["threshold"])
    test_s1 = pl.read_parquet(os.path.join(args.work, "test_norm.parquet"), columns=["entity_id", "src"]) \
        .filter(pl.col("src") == 1)["entity_id"].to_list()
    K = ["s1_id", "cand_id"]
    if CAND_P1_MIN is not None:
        cands = export_test_candidates(args.work, CAND_P1_MIN)
        n_before = matches.height
        matches = matches.join(cands.select(K), on=K, how="semi")
        if matches.height < n_before:
            log(f"  {n_before - matches.height:,} accepted matches dropped (pair below stage-1 candidate filter)")
    else:
        cands = pl.read_parquet(os.path.join(args.work, "test_cands.parquet"), columns=K)

    def to_map(df):
        """(s1_id, cand_id) pairs -> {s1_id: sorted list of cand_id}."""
        g = df.sort(["s1_id", "cand_id"]).group_by("s1_id", maintain_order=True).agg(pl.col("cand_id"))
        return dict(zip(g["s1_id"].to_list(), g["cand_id"].to_list()))

    os.makedirs(args.out, exist_ok=True)
    write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), test_s1, to_map(cands), "candidate_entity_ids")
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), test_s1, to_map(matches), "matched_entity_ids")
    n_with = matches["s1_id"].n_unique()
    log(f"done: {matches.height:,} matches for {n_with:,}/{len(test_s1):,} S1 entities -> {args.out}")


if __name__ == "__main__":
    main()
