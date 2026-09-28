"""099 / B6: India+US stage 3 with a third cross-encoder logit from a LARGER backbone (multilingual-e5-base, MIT).

The champion's stage 3 (v17r: stage-2 inputs + e5-small raw logit + its context + e5-small normalised-view logit) gets
p2_ce3_logit (e5-base, raw records, same zone / folds / recipe, cross-fitted) and its context (gap to the best other
candidate of the S1 and to the best other S1 claiming the record). Stage 3 is retrained with the production LightGBM
recipe (train.train_all, cross-fitted by S1 fold); the rank-rule thresholds are re-tuned out-of-fold at 1x.
2x corrected: the same frozen models on the 2x de-twinned pool (clones reuse the logit of their original record; the
missing pairs are scored with the e5-base fold models). v17r_nodba's EXP-005 exclusion is applied everywhere.
Countries without labels (France) are untouched.
usage: s3_ce3.py [--steps train,oof,2x,test]"""
import argparse
import json
import os
import sys
import time

import polars as pl

sys.path.insert(0, "code/business_entity_resolution/src")
import cross_encoder  # noqa: E402
from decide import assign_owner, keep_by_rank, tune_rank_thresholds  # noqa: E402
from stage2 import ce_context  # noqa: E402
from train import load_part, part_files, score_parts, train_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", default="train,oof,2x,test")
ap.add_argument("--ce", default="work/audit/099/ce_e5base")
ap.add_argument("--out", default="work/audit/099/s3ce3")
a = ap.parse_args()
steps = set(a.steps.split(","))
T0 = time.time()
K = ["s1_id", "cand_id"]
W, W2, M = "work_v17r", "work_v17r_2x", "model_v9"
cross_encoder.BACKBONE = "intfloat/multilingual-e5-base"
ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
os.makedirs(a.out, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {m}", flush=True)


def ce3_cols(ce, detwin=False):
    """(p2_ce3_logit, p2_ce3_gap_s1, p2_ce3_gap_cand) in the row order of ce (s1_id, cand_id, ce_logit)."""
    ce = ce.with_columns(pl.col("ce_logit").cast(pl.Float32))
    if detwin:  # context with each clone collapsed onto its original (identical text -> identical logit)
        C = ce.with_columns(ORIG.alias("_o")).group_by("s1_id", "_o").agg(pl.col("ce_logit").first()).rename({"_o": "cand_id"})
        cc = pl.concat([C.select("s1_id", pl.col("cand_id").alias("_o")), ce_context(C)], how="horizontal")
        ctx = ce.select("s1_id", ORIG.alias("_o")).join(cc, on=["s1_id", "_o"], how="left", maintain_order="left").drop("s1_id", "_o")
    else:
        ctx = ce_context(ce)
    return pl.concat([ce.select(*K, pl.col("ce_logit").alias("p2_ce3_logit")),
                      ctx.select(pl.col("p2_ce_gap_s1").alias("p2_ce3_gap_s1"), pl.col("p2_ce_gap_cand").alias("p2_ce3_gap_cand"))], how="horizontal")


def write_aligned(zone_dir, t, out_dir):
    """Parts of the new columns, row-aligned with the zone parts (every zone pair has a logit)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(part_files(zone_dir)):
        z = pl.read_parquet(f, columns=K).join(t, on=K, how="left", maintain_order="left")
        assert z["p2_ce3_logit"].null_count() == 0, f"missing ce3 logits in part {i}"
        z.drop(K).write_parquet(f"{out_dir}/part_{i:03d}.parquet")


def full_oof(p2_path, z, excl):
    return pl.read_parquet(p2_path, columns=[*K, "prob"]).join(z, on=K, how="left", maintain_order="left") \
        .with_columns(pl.coalesce("p3", "prob").alias("prob")).drop("p3").join(excl, on=K, how="anti")


def f05(oof, gt, top, rest):
    from evaluate import macro_f05
    s1 = oof["s1_id"].unique().to_list()
    pred = keep_by_rank(assign_owner(oof), top, rest).select(K)
    return macro_f05(pred, gt, pl.read_parquet(f"{W}/train_norm.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)["entity_id"].to_list())


gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").select(K)
zone1 = f"{W}/train_zone_ctx_ce2norm_{M}"
md = f"{a.out}/model"
if "train" in steps:
    tr = pl.read_parquet(f"{a.ce}/train_scores.parquet")
    log(f"e5-base train logits: {tr.height:,}")
    write_aligned(zone1, ce3_cols(tr), f"{a.out}/train_zone_ce3")
    train_all([zone1, f"{a.out}/train_zone_ce3"], md, prefix="stage3_ce3", log=log)
    log("stage 3 + ce3 trained")
m3 = None
if steps & {"oof", "2x", "test"}:
    from train import load_model, model_path
    m3 = [load_model(model_path(md, "stage3_ce3", k)) for k in (0, 1)]
thr_path = f"{a.out}/thresholds.json"
if "oof" in steps:
    excl = pl.read_parquet("work/audit/v17r/dba_only_train1x.parquet").select(K)
    z = score_parts([zone1, f"{a.out}/train_zone_ce3"], m3, oof=True, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    oof = full_oof(f"{W}/train_oof_{M}.parquet", z, excl)
    oof.write_parquet(f"{a.out}/train_oof_1x.parquet")
    champ = pl.read_parquet(f"work/audit/v17r/nodba_1x/train_oof_{M}_stage3_ctx_ce2norm.parquet", columns=[*K, "prob"])
    s1_all = pl.read_parquet(f"{W}/train_norm.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)["entity_id"].to_list()
    t_top, t_rest, s = tune_rank_thresholds(oof, gt, s1_all, log=lambda m: None)
    json.dump({"threshold_top": t_top, "threshold_rest": t_rest, "oof_macro_f05_rank": s}, open(thr_path, "w"))
    for name, o in (("champion", champ), ("stage3 + ce3", oof)):
        for tt, trr in ((0.55, 0.75), (t_top, t_rest)):
            m = f05(o, gt, tt, trr)
            log(f"1x {name:14s} rule {tt:.2f}/{trr:.2f}: F0.5 {m['macro_f05']:.5f} P {m['micro_precision']:.5f} R {m['micro_recall']:.5f}")
if "2x" in steps:
    thr = json.load(open(thr_path))
    zone2 = f"{W2}/train_zone_ctx_ce2norm"
    z2 = pl.concat([pl.read_parquet(f, columns=[*K, "fold"]) for f in part_files(zone2)])
    have = pl.read_parquet(f"{a.ce}/train_scores.parquet").rename({"cand_id": "_orig"})
    zz = z2.with_columns(ORIG.alias("_orig"))
    got = zz.join(have, on=["s1_id", "_orig"], how="inner")
    todo = zz.join(have, on=["s1_id", "_orig"], how="anti")
    log(f"2x zone {z2.height:,} pairs: {got.height:,} e5-base logits reused, {todo.height:,} to score")
    cache = f"{a.out}/ce3_2x_new.parquet"
    if todo.height and not os.path.exists(cache):
        new = cross_encoder.score_pairs("dataset", "train", a.ce, todo.select("s1_id", pl.col("_orig").alias("cand_id"), "fold").unique(["s1_id", "cand_id"]), logger=log)
        new.write_parquet(cache)
    ce2x = got.select(*K, "ce_logit")
    if todo.height:
        ce2x = pl.concat([ce2x, todo.join(pl.read_parquet(cache).rename({"cand_id": "_orig"}), on=["s1_id", "_orig"]).select(*K, "ce_logit")])
    write_aligned(zone2, ce3_cols(ce2x, detwin=True), f"{a.out}/train_zone_ce3_2x")
    z = score_parts([zone2, f"{a.out}/train_zone_ce3_2x"], m3, oof=True, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    excl2 = pl.read_parquet("work/audit/v17r/dba_only_train2x.parquet").select(K)
    oof2 = full_oof(f"{W2}/train_oof_{M}.parquet", z, excl2)
    oof2.write_parquet(f"{a.out}/train_oof_2x.parquet")
    champ2 = pl.read_parquet(f"work/audit/v17r/nodba_2x/train_oof_{M}_stage3_ctx_ce2norm.parquet", columns=[*K, "prob"])
    for name, o, tt, trr in (("champion", champ2, 0.55, 0.75), ("stage3 + ce3", oof2, thr["threshold_top"], thr["threshold_rest"]),
                             ("stage3 + ce3", oof2, 0.55, 0.75)):
        m = f05(o, gt, tt, trr)
        log(f"2x {name:14s} rule {tt:.2f}/{trr:.2f}: F0.5 {m['macro_f05']:.5f} P {m['micro_precision']:.5f} R {m['micro_recall']:.5f}")
if "test" in steps:
    zt = f"{W}/test_zone_ctx_ce2norm_{M}"
    te = pl.read_parquet(f"{a.ce}/test_scores.parquet")
    zk = pl.concat([pl.read_parquet(f, columns=K) for f in part_files(zt)])
    miss = zk.join(te, on=K, how="anti")
    if miss.height:
        log(f"test zone pairs without an e5-base logit: {miss.height:,} -> scoring")
        te = pl.concat([te, cross_encoder.score_pairs("dataset", "test", a.ce, miss, logger=log)])
    write_aligned(zt, ce3_cols(te.unique(K)), f"{a.out}/test_zone_ce3")
    p3 = score_parts([zt, f"{a.out}/test_zone_ce3"], m3, oof=False, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    p3.write_parquet(f"{a.out}/test_p3_ce3.parquet")
    log(f"test stage-3 (+ce3) probabilities for {p3.height:,} zone pairs")
log("DONE")
