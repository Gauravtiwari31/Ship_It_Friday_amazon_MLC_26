"""fr_claude M2 (production): France decision layer on the retrieval-feature unseen-country chain (work/audit/leader_gap/lg_frcascade_bi).
Train (India+US labels only): out-of-fold chain (stage-1/2 mean) + out-of-fold stage 3 on its zone → invariant group
features (same as loo_meta.py) → lambdarank owner ranker + binary risk model. Apply to France test (frozen; no test
fitting). Decide one-owner + --top/--rest. Writes work/fr_claude/fr_test_final_{tag}.parquet and France-only
submission/fr_claude/{matching,candidate}_{tag}_fr.tsv.
usage: fr_meta_prod.py --top T --rest R --tag NAME [--no-meta]"""
import os, sys, glob, json, time, argparse, hashlib
os.environ.setdefault("POLARS_MAX_THREADS", "16")
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "code/business_entity_resolution/src")
from decide import assign_owner, keep_by_rank
from train import load_model, part_files, score_parts
ap = argparse.ArgumentParser(); ap.add_argument("--top", type=float, required=True); ap.add_argument("--rest", type=float, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--no-meta", action="store_true"); a = ap.parse_args()
K = ["s1_id", "cand_id"]; C = "work/audit/leader_gap/lg_frcascade_bi"; O = "work/fr_claude"; T0 = time.time()
ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:.0f}s] {m}", flush=True)
m3 = [load_model(f"{C}/model/ustage3_fold{k}.txt") for k in (0, 1)]
def side(split):
    ch = pl.read_parquet(f"{C}/{split}_chain.parquet").select(*K, "p1", pl.col("prob").alias("chain"))
    zd = f"{C}/{split}_uzone"
    p3 = score_parts([zd], m3, oof=(split == "train"), log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    ce = pl.concat([pl.read_parquet(f, columns=[*K, "p2_ce_logit"]) for f in part_files(zd)])
    B = pl.read_parquet(f"work/audit/leader_gap/bienc_cands_{split}.parquet").select(*K, pl.lit(1.0).cast(pl.Float32).alias("f_bi_top1"),
        pl.col("bi_cos").cast(pl.Float32).alias("f_bi_cos1"), (pl.col("bi_cos") - pl.col("bi_cos2")).cast(pl.Float32).alias("f_bi_gap"))
    f = ch.join(p3, on=K, how="left").join(ce, on=K, how="left").with_columns(pl.coalesce("p3", "chain").alias("prob")) \
        .join(B, on=K, how="left").with_columns(pl.col("f_bi_top1").fill_null(0.0))
    log(f"{split}: {f.height:,} pairs, zone {f['p3'].is_not_null().sum():,}")
    return f
def meta_table(kind):
    nkx = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ")
    N3 = pl.read_parquet(f"work_lg/{kind}_norm.parquet", columns=["entity_id", "src", "country", "name_core", "addr_empty"])
    S1N = N3.filter(pl.col("src") == 1).with_columns(nkx.alias("nk")); S1N = S1N.join(S1N.group_by("country", "nk").len("name_owners"), on=["country", "nk"])
    return N3.filter(pl.col("src") != 1).with_columns(nkx.alias("nk")).join(S1N.select("country", "nk", "name_owners").unique(["country", "nk"]),
        on=["country", "nk"], how="left").select(pl.col("entity_id").alias("_o"), "addr_empty", pl.col("src").alias("csrc"), pl.col("name_owners").fill_null(0))
exec(open("code/business_entity_resolution/src/final/_mfeats.py", encoding="utf-8").read())  # mfeats(f, META), MF (shared with loo_meta)
tn = pl.read_parquet("work_lg/test_norm.parquet", columns=["entity_id", "src", "country"])
trc = set(pl.read_parquet("work_lg/train_norm.parquet", columns=["src", "country"]).filter(pl.col("src") == 1)["country"].unique().to_list())
UNSEEN = tn.filter((pl.col("src") == 1) & ~pl.col("country").is_in(list(trc)))["entity_id"]
ft = side("test").filter(pl.col("s1_id").is_in(UNSEEN.implode()))
if not a.no_meta:
    fs = side("train").join(pl.read_parquet("work_lg/train_gt_pairs.parquet").select(*K, pl.lit(1).cast(pl.Int8).alias("y")), on=K, how="left") \
        .with_columns(pl.col("y").fill_null(0))
    ds = mfeats(fs, meta_table("train")); log(f"meta training edges {ds.height:,}")
    PM = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=16, verbose=-1, seed=7)
    PR = dict(PM, objective="lambdarank", lambdarank_truncation_level=3)
    g_ = ds.filter((pl.col("r_n") >= 2) & (pl.col("y").sum().over("cand_id") >= 1)).sort("cand_id")
    rk = lgb.train(PR, lgb.Dataset(g_.select(MF).to_numpy().astype(np.float32), g_["y"].to_numpy(),
                   group=g_.group_by("cand_id", maintain_order=True).len()["len"].to_numpy(), feature_name=MF), 300)
    def addrk(d):
        return d.with_columns(pl.Series("rk", rk.predict(d.select(MF).to_numpy().astype(np.float32), num_threads=16))) \
            .with_columns(pl.col("rk").rank("ordinal", descending=True).over("cand_id").alias("rk_rank"), (pl.col("rk") - pl.col("rk").max().over("cand_id")).alias("rk_gap"))
    ds = addrk(ds); MF2 = MF + ["rk", "rk_rank", "rk_gap"]
    es = (ds["s1_id"].hash(5) % 20 == 0).to_numpy(); X = ds.select(MF2).to_numpy().astype(np.float32); y = ds["y"].to_numpy()
    bm = lgb.train(PM, lgb.Dataset(X[~es], y[~es], feature_name=MF2), 2000, valid_sets=[lgb.Dataset(X[es], y[es], feature_name=MF2)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    rk.save_model(f"{O}/meta_rank.txt"); bm.save_model(f"{O}/meta_bin.txt")
    dt = addrk(mfeats(ft, meta_table("test")))
    dt = dt.with_columns(pl.Series("q", bm.predict(dt.select(MF2).to_numpy().astype(np.float32), num_iteration=bm.best_iteration, num_threads=16)))
    ft = ft.join(dt.select(*K, "q"), on=K, how="left").with_columns(pl.coalesce("q", "prob").alias("prob")).drop("q")
fin = ft.select(*K, "prob"); fin.write_parquet(f"{O}/fr_test_final_{a.tag}.parquet")
m = keep_by_rank(assign_owner(fin), a.top, a.rest).select(K)
US_ = set(UNSEEN.to_list())
order = [s for s in pl.read_csv("dataset/test/test_source1.tsv", separator="\t", columns=["entity_id"], quote_char=None)["entity_id"].to_list() if s in US_]
def mp(df): return dict(df.sort(K).group_by("s1_id", maintain_order=True).agg("cand_id").iter_rows())
os.makedirs("submission/fr_claude", exist_ok=True)
for fn, col, d in ((f"candidate_{a.tag}_fr.tsv", "candidate_entity_ids", mp(fin.select(K))), (f"matching_{a.tag}_fr.tsv", "matched_entity_ids", mp(m))):
    with open(f"submission/fr_claude/{fn}", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in order: fh.write(f"{s}\t{','.join(d.get(s, []))}\n")
    log(f"{fn} sha256 {hashlib.sha256(open(f'submission/fr_claude/{fn}', 'rb').read()).hexdigest()}")
log(f"France S1 {len(order):,}; candidates {fin.height:,} ({fin.height/len(order):.3f}/S1); matches {m.height:,}; rule {a.top}/{a.rest}; meta {not a.no_meta}")
