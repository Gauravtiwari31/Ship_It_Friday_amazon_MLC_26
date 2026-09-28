"""Validation at the test's distractor density (Checkpoint 0).

The test has ~2.4 unmatched S2/S3 records per S1 vs 1.22 in train, and its excess records have the same
blocking-similarity profile as the train distractors (decoys of S1 entities). Simulation: every train distractor
record is cloned (factor - 1) times as a new record (same text, id + '#dK'), then blocking, features, both context
stages and the cross-encoder stage 3 are recomputed on this pool with the FROZEN production models / thresholds
(nothing is retrained; word-risk statistics are the original 1x fold statistics). Out-of-fold scores are written for
the original S1 set; evaluate with eval_oof.py.

usage: sim_density.py <work> <work_sim> <data> [--factor 2] [--steps all]"""
import argparse
import os
import shutil
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "code/business_entity_resolution/src")
import run_blocking  # noqa: E402
import run_features  # noqa: E402
from features_extra import add_rarity, diff_tokens, encode, rarity_tables, token_stats  # noqa: E402
from features_num import build_numrel  # noqa: E402
from stage2 import (ce_context, sibling_context, source_context, write_context_parts,  # noqa: E402
                    write_sibling_context_parts, write_source_context_parts)
from train import load_model, load_part, part_files, score_parts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("work"); ap.add_argument("sim"); ap.add_argument("data")
ap.add_argument("--factor", type=int, default=2)
ap.add_argument("--model", default="model_v9")
ap.add_argument("--steps", default="norm,block,feat,extra,numrel,s1,ctx,s2,ce,s3")
ap.add_argument("--frozen", default=None, help="load the models / cross-encoders from this snapshot instead of <work> "
                "(e.g. work/baselines/CHAMPION-v16/frozen), so retraining can never leak into an A/B comparison")
ap.add_argument("--detwin", action="store_true",
                help="a clone and its original do not see each other in the rarity counts and the sibling / same-source "
                     "context (real decoys of one S1 share an exact address only 13%% of the time; exact clones always do)")
a = ap.parse_args()
W, S, M = a.work, a.sim, a.model
steps = set(a.steps.split(","))
os.makedirs(S, exist_ok=True)
log = lambda m: print(m, flush=True)
orig_id = pl.col("cand_id").str.replace(r"#d\d+$", "")

if "norm" in steps:  # pool with (factor - 1) clones of every distractor record
    norm = pl.read_parquet(f"{W}/train_norm.parquet")
    gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet")
    dis = norm.filter(pl.col("src") != 1).join(gt.select(pl.col("cand_id").alias("entity_id")), on="entity_id", how="anti")
    clones = [dis.with_columns((pl.col("entity_id") + f"#d{k}").alias("entity_id")) for k in range(1, a.factor)]
    pl.concat([norm] + clones).write_parquet(f"{S}/train_norm.parquet")
    shutil.copy(f"{W}/train_gt_pairs.parquet", f"{S}/train_gt_pairs.parquet")
    log(f"pool: {norm.height:,} records + {sum(c.height for c in clones):,} clones of {dis.height:,} distractors")
    del norm, clones, dis
if "block" in steps:
    c = run_blocking.run(S, "train", log=log)
    log(f"sim candidates: {c.height:,}")
if "feat" in steps:
    run_features.run(S, "train", log=log)
feat = f"{S}/train_feats"
parts = part_files(feat)

if "extra" in steps:  # rarity from the sim pool; word-risk encoded with the ORIGINAL 1x out-of-fold statistics
    out_dir = f"{S}/train_extra"
    os.makedirs(out_dir, exist_ok=True)
    names0 = pl.read_parquet(f"{W}/train_norm.parquet", columns=["entity_id", "name_core"])
    def with_names(p, names):
        return (p.join(names.rename({"entity_id": "s1_id", "name_core": "name_core_a"}), on="s1_id", how="left")
                .join(names.rename({"entity_id": "cand_id", "name_core": "name_core_b"}), on="cand_id", how="left"))
    tok = pl.concat([diff_tokens(with_names(pl.read_parquet(f, columns=["s1_id", "cand_id", "label", "fold"]), names0))
                     .select("label", "fold", "extra", "missing") for f in part_files(f"{W}/train_feats")])
    fold_stats = {k: token_stats(tok.filter(pl.col("fold") == k)) for k in (0, 1)}
    fold_prior = {k: tok.filter(pl.col("fold") == k)["label"].mean() for k in (0, 1)}
    del tok
    snorm = pl.read_parquet(f"{S}/train_norm.parquet", columns=["entity_id", "src", "country", "name_core", "addr_tokens", "addr_nums"])
    if a.detwin:  # rarity counted over original records; a clone gets its original's counts
        is_clone = pl.col("entity_id").str.contains("#d")
        rar = rarity_tables(snorm.filter(~is_clone))
        cl = snorm.filter(is_clone).select("entity_id", pl.col("entity_id").str.replace(r"#d\d+$", "").alias("_o"))
        rar = pl.concat([rar, cl.join(rar.rename({"entity_id": "_o"}), on="_o").select(rar.columns)])
    else:
        rar = rarity_tables(snorm)
    names = snorm.select("entity_id", "name_core")
    del snorm
    for i, f in enumerate(parts):
        d = diff_tokens(with_names(pl.read_parquet(f, columns=["s1_id", "cand_id", "fold"]), names))
        enc = [encode(d.with_row_index("_ord").filter(pl.col("fold") == k), fold_stats[1 - k], fold_prior[1 - k]) for k in (0, 1)]
        d = add_rarity(pl.concat(enc).sort("_ord").drop("_ord"), rar)
        cols = [c for c in d.columns if c.startswith("f_")]
        d.select([pl.col(c).cast(pl.Float32) for c in cols]).write_parquet(f"{out_dir}/part_{i:03d}.parquet")
    log("extra features done")
if "numrel" in steps:
    build_numrel(S, "train", parts, log=lambda m: None)
    log("numrel done")

base = [feat, f"{S}/train_extra", f"{S}/train_numrel"]
MD = a.frozen or W  # models and cross-encoder folds
md = f"{MD}/{M}"
if "s1" in steps or "ctx" in steps:
    m1 = [load_model(f"{md}/stage1_fold{k}.txt") for k in (0, 1)]
    sc = score_parts(base, m1, oof=True, log=lambda m: None)
    sc.select("s1_id", "cand_id", "prob").write_parquet(f"{S}/train_oof_stage1.parquet")
    sizes = [pl.scan_parquet(f).select(pl.len()).collect().item() for f in parts]
    write_context_parts(sc, sizes, f"{S}/train_p1ctx")
    cand_src = pl.read_parquet(f"{S}/train_norm.parquet", columns=["entity_id", "src"])
    attr = pl.read_parquet(f"{S}/train_norm.parquet", columns=["entity_id", "addr_tokens", "addr_nums", "name_core"])
    if a.detwin:  # sibling / same-source context computed with each clone collapsed onto its original record
        def detwinned(fn, arg, out_dir):
            C = sc.with_columns(orig_id.alias("_o")).group_by("s1_id", "_o").agg(pl.col("prob").max()).rename({"_o": "cand_id"})
            ctx = pl.concat([C.select("s1_id", pl.col("cand_id").alias("_o")), fn(C, arg)], how="horizontal")
            full = sc.select("s1_id", orig_id.alias("_o")).join(ctx, on=["s1_id", "_o"], how="left", maintain_order="left").drop("s1_id", "_o")
            os.makedirs(out_dir, exist_ok=True)
            offs = np.concatenate([[0], np.cumsum(sizes)])
            for i in range(len(sizes)):
                full.slice(int(offs[i]), int(sizes[i])).write_parquet(f"{out_dir}/part_{i:03d}.parquet")
        detwinned(source_context, cand_src, f"{S}/train_srcctx")
        detwinned(sibling_context, attr, f"{S}/train_sibctx")
    else:
        write_source_context_parts(sc, cand_src, sizes, f"{S}/train_srcctx")
        write_sibling_context_parts(sc, attr, sizes, f"{S}/train_sibctx")
    del sc
    log("stage-1 scores + context done")
dirs2 = base + [f"{S}/train_p1ctx", f"{S}/train_srcctx", f"{S}/train_sibctx"]
if "s2" in steps:
    m2 = [load_model(f"{md}/stage2_fold{k}.txt") for k in (0, 1)]
    score_parts(dirs2, m2, oof=True, log=lambda m: None).write_parquet(f"{S}/train_oof_{M}.parquet")
    log("stage-2 scores done")


def ce_logits(view_dir, view, zone):
    """Logits of the sim's uncertain pairs: reuse the production out-of-fold logit of (s1, original record) when it
    exists (a clone has the same text as its original), score the rest with the same fold models (GPU)."""
    import cross_encoder
    have = pl.read_parquet(f"{view_dir}/train_scores.parquet").rename({"cand_id": "_orig"})
    z = zone.with_columns(orig_id.alias("_orig"))
    got = z.join(have, on=["s1_id", "_orig"], how="inner")
    todo = z.join(have, on=["s1_id", "_orig"], how="anti")
    log(f"  {view}: {got.height:,} logits reused, {todo.height:,} to score")
    new = cross_encoder.score_pairs(a.data, "train", view_dir, todo.select("s1_id", pl.col("_orig").alias("cand_id"), "fold"),
                                    logger=log, view=view, work=W) if todo.height else None
    out = got.select("s1_id", "cand_id", "ce_logit")
    if new is not None:
        out = pl.concat([out, todo.select("s1_id", "cand_id").with_columns(new["ce_logit"])])
    return out


for view, sub in (("raw", ""), ("norm", "_norm")):
    if "ce" in steps or f"ce_{view}" in steps:
        p2 = pl.read_parquet(f"{S}/train_oof_{M}.parquet", columns=["s1_id", "cand_id", "prob"])
        folds = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]) for f in parts]).unique("s1_id")
        zone = p2.filter((pl.col("prob") > 0.01) & (pl.col("prob") < 0.99)).join(folds, on="s1_id").select("s1_id", "cand_id", "fold")
        log(f"uncertain pairs: {zone.height:,}")
        ce_logits(f"{MD}/ce_{M}{sub}", view, zone).write_parquet(f"{S}/ce_{view}.parquet")

for version, (tag, ctx, ce2) in (("v10", ("", False, False)), ("v16", ("_ctx_ce2norm", True, True))):  # stage 3 of v10 / v16
    if not ("s3" in steps or f"s3_{version}" in steps):
        continue
    p2 = pl.read_parquet(f"{S}/train_oof_{M}.parquet", columns=["s1_id", "cand_id", "prob"])
    raw = pl.read_parquet(f"{S}/ce_raw.parquet")
    nrm = pl.read_parquet(f"{S}/ce_norm.parquet").select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce2_logit")) if ce2 else None
    if True:
        if ctx and a.detwin:  # cross-encoder context with each clone collapsed onto its original (identical logit)
            C = raw.with_columns(orig_id.alias("_o")).group_by("s1_id", "_o").agg(pl.col("ce_logit").first()).rename({"_o": "cand_id"})
            cc = pl.concat([C.select("s1_id", pl.col("cand_id").alias("_o")), ce_context(C)], how="horizontal")
            ctx_cols = [raw.select("s1_id", orig_id.alias("_o")).join(cc, on=["s1_id", "_o"], how="left", maintain_order="left").drop("s1_id", "_o")]
        else:
            ctx_cols = [ce_context(raw)] if ctx else []
        t = pl.concat([raw.select("s1_id", "cand_id", pl.col("ce_logit").cast(pl.Float32).alias("p2_ce_logit"))] + ctx_cols,
                      how="horizontal")
        if ce2:
            t = t.join(nrm, on=["s1_id", "cand_id"], how="left", maintain_order="left")
        zd = f"{S}/train_zone{tag}"
        os.makedirs(zd, exist_ok=True)
        for i in range(len(parts)):
            load_part(i, dirs2).join(t, on=["s1_id", "cand_id"], how="inner", maintain_order="left").write_parquet(f"{zd}/part_{i:03d}.parquet")
        m3 = [load_model(f"{md}/stage3{tag}_fold{k}.txt") for k in (0, 1)]
        z = score_parts([zd], m3, oof=True, log=lambda m: None).select("s1_id", "cand_id", pl.col("prob").alias("p3"))
        p2.join(z, on=["s1_id", "cand_id"], how="left", maintain_order="left").with_columns(pl.coalesce("p3", "prob").alias("prob")) \
            .drop("p3").write_parquet(f"{S}/train_oof_{M}_stage3{tag}.parquet")
        log(f"stage 3{tag or ' (v10)'} done")
