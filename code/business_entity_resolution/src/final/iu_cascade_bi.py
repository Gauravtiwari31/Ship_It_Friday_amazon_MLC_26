"""iu_claude IU-1: the India+US cascade of lg_bienc_full_v2 + learned-retriever features (f_bi_top1, f_bi_cos1, f_bi_gap:
is the pair its record's top-1 bi-encoder retrieval, cosine, margin to the runner-up; out-of-fold on train, clones inherit
their original's values) in ALL stages. Stage 1 is retrained with them (it also drives the p1 >= tau candidate filter).
Original: leader_gap C1: compact candidate cascade for India+US (organizers rank smaller candidate sets higher).

candidate generation = TF-IDF blocking (unchanged) -> learned pairwise filter: stage-1 LightGBM probability >= tau
(cross-fitted OOF on train, frozen fold models on 2x / test). candidate_pairs = the filtered set.
matcher = stage 2 with contexts RECOMPUTED over the filtered pairs only (retrained, cross-fitted) -> stage 3 on its zone
(0.01<p2<0.99) with the three cross-encoder logits + context (retrained) -> one-owner + rank rule (re-tuned on 1x).
Base chain: work_v17r (v19_dba_ce3 production features). 2x = corrected density (de-twinned source/sibling/CE context).
Steps: filt (filter + contexts, train/x2/test), s2, ce (GPU: logits for zone pairs without one), s3.
Writes only to --out."""
import argparse, json, os, sys, time, glob
os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np, polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
import train as TR
TR.LGB_PARAMS["num_threads"] = 12
from decide import assign_owner, keep_by_rank
from evaluate import macro_f05
from stage2 import ce_context, sibling_context, source_context, write_context_parts, write_sibling_context_parts, write_source_context_parts
from train import load_model, load_part, model_path, part_files, score_parts, train_all

ap = argparse.ArgumentParser()
ap.add_argument("--steps", default="bi,s1,filt,s2,ce,s3"); ap.add_argument("--tau", type=float, default=0.003)
ap.add_argument("--out", default="work/iu_claude/iu_cascade_bi")
ap.add_argument("--work", default="work_v17r"); ap.add_argument("--no-x2", action="store_true")
ap.add_argument("--work2", default="work_v17r_2x")
ap.add_argument("--bicos", default="", help="bi-encoder cosine parquet prefix (…/bicos_iu) -> stage-3 feature, outputs *_bc")
a = ap.parse_args(); steps = set(a.steps.split(","))
T0 = time.time(); K = ["s1_id", "cand_id"]; W, W2, O = a.work, a.work2, a.out; MD = f"{O}/model"
SPLITS = ("train", "test") if a.no_x2 else ("train", "x2", "test")
WV = "work_v17r"  # cross-encoder fold models / cached logits (pair-keyed: same entity ids in every work folder)
os.makedirs(MD, exist_ok=True)
ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:5.0f}s] {m}", flush=True)
BASE0 = {"train": [f"{W}/train_feats", f"{W}/train_extra", f"{W}/train_numrel"],
         "x2": [f"{W2}/train_feats", f"{W2}/train_extra", f"{W2}/train_numrel"],
         "test": [f"{W}/test_feats", f"{W}/test_extra", f"{W}/test_numrel"]}
BASE = {s: BASE0[s] + [f"{O}/{s}_bi"] for s in BASE0}
P1 = {s: f"{O}/{s}_p1new" for s in BASE0}
NORM = {"train": f"{W}/train_norm.parquet", "x2": f"{W2}/train_norm.parquet", "test": f"{W}/test_norm.parquet"}
FB = {s: f"{O}/{s}_base" for s in BASE}
CTX = {s: [f"{O}/{s}_p1ctx", f"{O}/{s}_srcctx", f"{O}/{s}_sibctx"] for s in BASE}
gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").select(K)
s1_train = pl.read_parquet(f"{W}/train_norm.parquet", columns=["entity_id", "src"]).filter(pl.col("src") == 1)["entity_id"].to_list()
tn = pl.read_parquet(f"{W}/test_norm.parquet", columns=["entity_id", "src", "country"])
IU_TEST = tn.filter((pl.col("src") == 1) & pl.col("country").is_in(["India", "US"]))["entity_id"]

def sizes(d): return [pl.scan_parquet(f).select(pl.len()).collect().item() for f in part_files(d)]
if "bi" in steps:  # retriever features, row-aligned with the base parts (2x clones: their original record's retrieval)
    for split in SPLITS:
        B = pl.read_parquet(f"work/audit/leader_gap/bienc_cands_{'test' if split == 'test' else 'train'}.parquet").select(
            *K, pl.lit(1.0).cast(pl.Float32).alias("f_bi_top1"), pl.col("bi_cos").cast(pl.Float32).alias("f_bi_cos1"),
            (pl.col("bi_cos") - pl.col("bi_cos2")).cast(pl.Float32).alias("f_bi_gap")).rename({"cand_id": "_o"})
        os.makedirs(f"{O}/{split}_bi", exist_ok=True)
        for i, f in enumerate(part_files(BASE0[split][0])):
            pl.read_parquet(f, columns=K).with_columns(ORIG.alias("_o")).join(B, on=["s1_id", "_o"], how="left", maintain_order="left")                 .with_columns(pl.col("f_bi_top1").fill_null(0.0)).select("f_bi_top1", "f_bi_cos1", "f_bi_gap").write_parquet(f"{O}/{split}_bi/part_{i:03d}.parquet")
        log(f"{split}: retriever features written")
if "s1" in steps:  # production stage-1 features (incl. word-risk) + retriever features; OOF on train / 2x, fold mean on test
    m1 = train_all(BASE["train"], MD, prefix="stage1bi", log=log)
    for split in SPLITS:
        sc = score_parts(BASE[split], m1, oof=(split != "test"), log=lambda m: None)
        sz = sizes(BASE0[split][0]); offs = np.concatenate([[0], np.cumsum(sz)]); os.makedirs(P1[split], exist_ok=True)
        for j in range(len(sz)):
            sc.slice(int(offs[j]), int(sz[j])).select(pl.col("prob").alias("p1_prob")).write_parquet(f"{P1[split]}/part_{j:03d}.parquet")
        log(f"{split}: new stage-1 probabilities written")

if "filt" in steps:
    for split in SPLITS:
        os.makedirs(FB[split], exist_ok=True); scs = []; n0 = n1 = 0
        for i in range(len(part_files(BASE[split][0]))):
            df = load_part(i, BASE[split]); p1 = pl.read_parquet(part_files(P1[split])[i], columns=["p1_prob"])["p1_prob"]
            assert len(p1) == df.height
            keep = (p1 >= a.tau)
            if split == "test":
                keep = keep & df["s1_id"].is_in(IU_TEST.implode())
            n0 += df.height; df = df.filter(keep); n1 += df.height
            df.write_parquet(f"{FB[split]}/part_{i:03d}.parquet")
            scs.append(df.select(*K).with_columns(p1.filter(keep).alias("prob")))
        sc = pl.concat(scs); sz = sizes(FB[split])
        log(f"{split}: kept {n1:,} of {n0:,} pairs (tau {a.tau})")
        norm = pl.read_parquet(NORM[split], columns=["entity_id", "src", "addr_tokens", "addr_nums", "name_core"])
        write_context_parts(sc, sz, CTX[split][0])
        if split != "x2":
            write_source_context_parts(sc, norm.select("entity_id", "src"), sz, CTX[split][1])
            write_sibling_context_parts(sc, norm.filter(pl.col("src") != 1).select("entity_id", "addr_tokens", "addr_nums", "name_core"), sz, CTX[split][2])
        else:  # de-twinned source / sibling context (as reports/ceiling6h/l2_cached_chain.py)
            canonical = sc.with_columns(ORIG.alias("cand_id")).group_by(K).agg(pl.col("prob").max())
            offs = np.concatenate([[0], np.cumsum(sz)])
            for fn, cols, out in ((source_context, ["entity_id", "src"], CTX[split][1]),
                                  (sibling_context, ["entity_id", "addr_tokens", "addr_nums", "name_core"], CTX[split][2])):
                arg = norm.select(cols) if fn is source_context else norm.filter(pl.col("src") != 1).select(cols)
                c = pl.concat([canonical.select(K), fn(canonical, arg)], how="horizontal").rename({"cand_id": "_o"})
                full = sc.select("s1_id", ORIG.alias("_o")).join(c, on=["s1_id", "_o"], how="left", maintain_order="left").drop("s1_id", "_o")
                os.makedirs(out, exist_ok=True)
                for j in range(len(sz)): full.slice(int(offs[j]), int(sz[j])).write_parquet(f"{out}/part_{j:03d}.parquet")
        log(f"{split}: contexts written")
        del sc, scs, norm

DIRS = {s: [FB[s]] + CTX[s] for s in BASE}
if "s2" in steps:
    m2 = train_all(DIRS["train"], MD, prefix="stage2", log=log)
    score_parts(DIRS["train"], m2, oof=True, log=lambda m: None).select(*K, "prob").write_parquet(f"{O}/train_s2.parquet")
    if "x2" in SPLITS:
        score_parts(DIRS["x2"], m2, oof=True, log=lambda m: None).select(*K, "prob").write_parquet(f"{O}/x2_s2.parquet")
    score_parts(DIRS["test"], m2, oof=False, log=lambda m: None).select(*K, "prob").write_parquet(f"{O}/test_s2.parquet")
    log("stage 2 done")

CE = {"raw": (f"{WV}/ce_model_v9", "raw", "intfloat/multilingual-e5-small"),
      "norm": (f"{WV}/ce_model_v9_norm", "norm", "intfloat/multilingual-e5-small"),
      "e5b": ("work/audit/099/ce_e5base", "raw", "intfloat/multilingual-e5-base")}
HAVE = {("raw", "train"): ["work_v17r_2x/ce_raw.parquet", "work/audit/099/l2/ce_raw_train_new.parquet", "work/audit/c6b/ce_fill/raw_train.parquet"],
        ("norm", "train"): ["work_v17r_2x/ce_norm.parquet", "work/audit/099/l2/ce_norm_train_new.parquet", "work/audit/c6b/ce_fill/norm_train.parquet"],
        ("e5b", "train"): ["work/audit/099/s3ce3/ce3_2x_new.parquet", "work/audit/099/l2/ce_e5b_train_new.parquet", "work/audit/c6b/ce_fill/e5b_train.parquet"],
        ("raw", "test"): ["work/audit/099/ce_raw_test_extra.parquet", "work/audit/099/l2/ce_raw_test_new.parquet", "work/audit/c6b/ce_fill/raw_test.parquet"],
        ("norm", "test"): ["work/audit/099/ce_norm_test_extra.parquet", "work/audit/099/l2/ce_norm_test_new.parquet", "work/audit/c6b/ce_fill/norm_test.parquet"],
        ("e5b", "test"): ["work/audit/099/l2/ce_e5b_test_new.parquet", "work/audit/c6b/ce_fill/e5b_test.parquet"]}
def zone(p): return pl.read_parquet(p).filter((pl.col("prob") > .01) & (pl.col("prob") < .99)).select(K)
def have(name, sp):
    d = CE[name][0]
    prev = [f"work/audit/leader_gap/{r}/ce_{name}_{sp}_new.parquet" for r in ("cascade_t003", "lg_cascade")]  # earlier runs
    fs = [f"{d}/{sp}_scores.parquet"] + [f for f in HAVE[(name, sp)] + prev if os.path.exists(f)] \
        + [f"{O}/ce_{name}_{sp}_new.parquet"] * os.path.exists(f"{O}/ce_{name}_{sp}_new.parquet")
    return pl.concat([pl.read_parquet(f).select(*K, pl.col("ce_logit").cast(pl.Float32)).with_columns(ORIG.alias("cand_id")) for f in fs]).unique(K)
if "ce" in steps:
    import cross_encoder
    folds = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]) for f in part_files(f"{W}/train_feats")]).unique("s1_id")
    ztr = pl.concat([zone(f"{O}/train_s2.parquet")] + ([zone(f"{O}/x2_s2.parquet")] if "x2" in SPLITS else [])).with_columns(ORIG.alias("cand_id")).unique(K).join(folds, on="s1_id")
    zte = zone(f"{O}/test_s2.parquet")
    for name, (d, view, bb) in CE.items():
        for sp, z in (("train", ztr), ("test", zte)):
            todo = z.join(have(name, sp), on=K, how="anti")
            log(f"{name} {sp}: {z.height:,} zone pairs, {todo.height:,} without a logit")
            if todo.height:
                cross_encoder.BACKBONE = bb
                new = cross_encoder.score_pairs("dataset", sp, d, todo, logger=log, view=view, work=WV).with_columns(pl.col("ce_logit").cast(pl.Float32))
                new.write_parquet(f"{O}/ce_{name}_{sp}_new.parquet")

def zone_table(split):
    sp = "test" if split == "test" else "train"
    z = pl.read_parquet(f"{O}/{split}_s2.parquet").filter((pl.col("prob") > .01) & (pl.col("prob") < .99)).select(K)
    out = z
    for name, col in (("raw", "p2_ce_logit"), ("norm", "p2_ce2_logit"), ("e5b", "p2_ce3_logit")):
        h = have(name, sp).rename({"cand_id": "_o"})
        t = z.with_columns(ORIG.alias("_o")).join(h, on=["s1_id", "_o"], how="left", maintain_order="left")
        miss = t["ce_logit"].null_count()
        if miss: log(f"  {split} {name}: {miss:,} zone pairs still without a logit (NaN)")
        v = t.filter(pl.col("ce_logit").is_not_null())
        if split == "x2":
            C = v.group_by("s1_id", "_o").agg(pl.col("ce_logit").first()).rename({"_o": "cand_id"})
            cc = pl.concat([C.select("s1_id", pl.col("cand_id").alias("_o")), ce_context(C)], how="horizontal")
            ctx = v.select(*K, "_o").join(cc, on=["s1_id", "_o"], how="left", maintain_order="left").drop("_o")
        else:
            ctx = pl.concat([v.select(K), ce_context(v.select(*K, "ce_logit"))], how="horizontal")
        t = t.select(*K, pl.col("ce_logit").alias(col))
        if name == "raw":
            t = t.join(ctx, on=K, how="left")
        elif name == "e5b":
            t = t.join(ctx.select(*K, pl.col("p2_ce_gap_s1").alias("p2_ce3_gap_s1"), pl.col("p2_ce_gap_cand").alias("p2_ce3_gap_cand")), on=K, how="left")
        out = out.join(t, on=K, how="left")
    if a.bicos:
        bc = pl.read_parquet(f"{a.bicos}_{sp}.parquet").select(*K, pl.col("bi_cos").cast(pl.Float32).alias("p2_bi_cos"))
        out = out.join(bc, on=K, how="left")
    return out
SFX = "_bc" if a.bicos else ""
def write_zone(split):
    zd = f"{O}/{split}_zone{SFX}"; os.makedirs(zd, exist_ok=True); t = zone_table(split)
    for i in range(len(part_files(DIRS[split][0]))):
        load_part(i, DIRS[split]).join(t, on=K, how="inner", maintain_order="left").write_parquet(f"{zd}/part_{i:03d}.parquet")
    return zd
def final(split, m3, oof):
    zd = write_zone(split)
    z = score_parts([zd], m3, oof=oof, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    return pl.read_parquet(f"{O}/{split}_s2.parquet").join(z, on=K, how="left", maintain_order="left") \
        .with_columns(pl.coalesce("p3", "prob").alias("prob")).drop("p3")
def metr(o, t, r):
    m = macro_f05(keep_by_rank(assign_owner(o), t, r).select(K), gt, s1_train)
    pred = keep_by_rank(assign_owner(o), t, r).select(K); tp = pred.join(gt, on=K, how="semi").height
    return dict(F05=round(m["macro_f05"], 6), P=round(m["micro_precision"], 6), R=round(m["micro_recall"], 6), FP=pred.height - tp, FN=gt.height - tp)
if "s3" in steps:
    zd = write_zone("train")
    m3 = train_all([zd], MD, prefix="stage3" + SFX, log=log)
    o1 = score_parts([zd], m3, oof=True, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    o1 = pl.read_parquet(f"{O}/train_s2.parquet").join(o1, on=K, how="left", maintain_order="left").with_columns(pl.coalesce("p3", "prob").alias("prob")).drop("p3")
    o1.write_parquet(f"{O}/train_final{SFX}.parquet")
    best = (None, -1)
    owned = assign_owner(o1)
    for t in (0.5, 0.55, 0.6, 0.65):
        for r in (0.65, 0.7, 0.75, 0.8):
            f = macro_f05(keep_by_rank(owned, t, r).select(K), gt, s1_train)["macro_f05"]
            if f > best[1]: best = ((t, r), f)
    rule = best[0]; res = {"tau": a.tau, "rule": rule}
    res["1x@.60/.75"] = metr(o1, .60, .75); res["1x@tuned"] = metr(o1, *rule)
    res["cand_per_s1_train"] = round(pl.read_parquet(f"{O}/train_s2.parquet", columns=["s1_id"]).height / len(s1_train), 3)
    log(json.dumps(res))
    if "x2" in SPLITS:
        o2 = final("x2", m3, oof=True); o2.write_parquet(f"{O}/x2_final{SFX}.parquet")
        res["2x@.60/.75"] = metr(o2, .60, .75); res["2x@tuned"] = metr(o2, *rule); log(json.dumps(res))
    ot = final("test", m3, oof=False); ot.write_parquet(f"{O}/test_final_iu{SFX}.parquet")
    res["test_iu_pairs"] = ot.height; res["test_cand_per_iu_s1"] = round(ot.height / len(IU_TEST), 3)
    json.dump(res, open(f"{O}/metrics{SFX}.json", "w"), indent=1); log(json.dumps(res)); log("DONE")
