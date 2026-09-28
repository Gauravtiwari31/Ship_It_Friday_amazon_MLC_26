"""iu_claude IU-3/4 (on the IU-1 chain work/iu_claude/iu_cascade_bi; + retriever features). Original: leader_gap R3/R4/R8: owner/group ranker + edge meta model (FP risk) + ambiguous-bucket confirmation. CPU only.

Input: champion (v19_dba_ce3 chain) out-of-fold final probabilities, 1x and corrected 2x
(work/audit/ceiling6h/retrieval/champion_oof_{1x,2x}.parquet). Population: edges with prob >= 0.01 (others can never be
accepted). Features are computed from OOF predictions only (never labels):
  edge: prob, stage-2 prob, CE logits; record group: owner count, rank, best/second, margin; S1 group: rank, sums,
  confident siblings (all / same source / other source); competitor (best other owner of the record) edge features;
  entity prototype: record raw name / address vs the S1's confident siblings (rapidfuzz max), same for the competitor;
  record meta: empty address, #S1 sharing the name key, source.
Models (LightGBM, 2 folds by hash(record) -> every edge scored out-of-fold):
  R3 lambdarank over record groups (>= 2 owners) -> owner choice;  R4 binary edge model (+ ranker score) -> new prob.
Decision: one-owner by new prob + rank rule re-tuned on 1x; the 2x set is scored by the 1x models (frozen).
"""
import os, sys, time, glob, json, argparse
os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import process, fuzz
sys.path.insert(0, "code/business_entity_resolution/src")
from decide import assign_owner, keep_by_rank
from evaluate import macro_f05

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=12); a = ap.parse_args()
K = ["s1_id", "cand_id"]; R = "work/iu_claude/iu_cascade_bi"; O = "work/iu_claude/owner_rank3"; os.makedirs(O, exist_ok=True)
ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
T0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:.0f}s] {m}", flush=True)

gt = pl.read_parquet("work_lg/train_gt_pairs.parquet").select(K)
def prep(kind):
    raw = pl.read_parquet(f"work_v17r/{kind}_raw.parquet", columns=["entity_id", "business_name", "business_address", "src"])         .with_columns(pl.col("business_name").fill_null("").str.to_lowercase().alias("nm"), pl.col("business_address").fill_null("").str.to_lowercase().alias("ad"))
    norm = pl.read_parquet(f"work_v17r/{kind}_norm.parquet", columns=["entity_id", "src", "country", "name_core", "addr_empty"])
    nk = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ")
    s1n = norm.filter(pl.col("src") == 1).with_columns(nk.alias("nk"))
    s1n = s1n.join(s1n.group_by("country", "nk").len("name_owners"), on=["country", "nk"])
    rec_nk = norm.filter(pl.col("src") != 1).with_columns(nk.alias("nk")).select(pl.col("entity_id").alias("_o"), "country", "nk", "addr_empty", pl.col("src").alias("csrc"))
    rm = rec_nk.join(s1n.select("country", "nk", "name_owners").unique(["country", "nk"]), on=["country", "nk"], how="left")         .with_columns(pl.col("name_owners").fill_null(0)).select("_o", "addr_empty", "csrc", "name_owners")
    return rm, s1n["entity_id"].to_list(), raw.select(pl.col("entity_id").alias("_id"), "nm", "ad")
PREP = {"train": prep("train")}
rec_meta, S1_IDS, TXT = PREP["train"]
FIN = {"1x": "train_final", "2x": "x2_final", "test": "test_final_iu"}; S2F = {"1x": "train_s2", "2x": "x2_s2", "test": "test_s2"}

def ce_table(split):  # stage-2 probability of the IU-1 chain + learned-retriever features (clones: their original's)
    out = pl.read_parquet(f"{R}/{S2F[split]}.parquet", columns=[*K, "prob"]).rename({"prob": "p2"})
    B = pl.read_parquet(f"work/audit/leader_gap/bienc_cands_{'test' if split == 'test' else 'train'}.parquet").select(pl.col("s1_id"), pl.col("cand_id").alias("_o"),
        pl.lit(1.0).alias("bi_top1"), pl.col("bi_cos"), (pl.col("bi_cos") - pl.col("bi_cos2")).alias("bi_gap"))
    return out.with_columns(ORIG.alias("_o")).join(B, on=["s1_id", "_o"], how="left").with_columns(pl.col("bi_top1").fill_null(0.0)).drop("_o")

def features(split):
    t = time.time()
    rec_meta, _, TXT = PREP["test" if split == "test" else "train"]
    d = pl.read_parquet(f"{R}/{FIN[split]}.parquet").filter(pl.col("prob") >= 0.01).with_columns(ORIG.alias("_o"))
    log(f"{split}: population {d.height:,} edges")
    d = d.join(ce_table(split), on=K, how="left")
    lp = (pl.col("prob").clip(1e-6, 1 - 1e-6) / (1 - pl.col("prob").clip(1e-6, 1 - 1e-6))).log()
    d = d.with_columns(lp.alias("lp"),
        pl.len().over("cand_id").alias("r_n"), pl.col("prob").rank("ordinal", descending=True).over("cand_id").alias("r_rank"),
        pl.col("prob").max().over("cand_id").alias("r_best"), pl.col("prob").sum().over("cand_id").alias("r_sum"),
        pl.len().over("s1_id").alias("s_n"), pl.col("prob").rank("ordinal", descending=True).over("s1_id").alias("s_rank"),
        pl.col("prob").sum().over("s1_id").alias("s_sum"), (pl.col("prob") >= 0.5).sum().over("s1_id").alias("s_n05"))
    d = d.with_columns(pl.col("prob").top_k(2).get(1, null_on_oob=True).over("cand_id").fill_null(0).alias("r_second")) \
        .with_columns(pl.when(pl.col("r_rank") == 1).then(pl.col("prob") - pl.col("r_second")).otherwise(pl.col("prob") - pl.col("r_best")).alias("r_margin"))
    d = d.join(rec_meta, on="_o", how="left")
    # confident siblings of each S1 (prob >= 0.9 and record's best owner), by source
    conf = d.filter((pl.col("prob") >= 0.9) & (pl.col("r_rank") == 1)).select("s1_id", "cand_id", "_o", "csrc")
    cs = conf.group_by("s1_id").agg(pl.len().alias("sib_n"), (pl.col("csrc") == 2).sum().alias("sib_n2"), (pl.col("csrc") == 3).sum().alias("sib_n3"))
    d = d.join(cs, on="s1_id", how="left").with_columns(pl.col(["sib_n", "sib_n2", "sib_n3"]).fill_null(0))
    self_conf = ((pl.col("prob") >= 0.9) & (pl.col("r_rank") == 1)).cast(pl.Int32)
    d = d.with_columns((pl.col("sib_n") - self_conf).alias("sib_n"),
                       pl.when(pl.col("csrc") == 2).then(pl.col("sib_n2") - self_conf).otherwise(pl.col("sib_n3") - self_conf).alias("sib_same"),
                       pl.when(pl.col("csrc") == 2).then(pl.col("sib_n3")).otherwise(pl.col("sib_n2")).alias("sib_other")).drop("sib_n2", "sib_n3")
    # prototype similarity: record vs S1's confident siblings (excluding itself); record vs S1 raw
    # only where a decision is open: multi-owner records or not-yet-certain edges (memory: ~3 siblings per edge)
    ex = d.filter((pl.col("r_n") >= 2) | (pl.col("prob") < 0.995)).select("s1_id", "cand_id", "_o") \
        .join(conf.select("s1_id", pl.col("_o").alias("_sib")), on="s1_id").filter(pl.col("_sib") != pl.col("_o"))
    ex = ex.join(TXT.rename({"_id": "_o", "nm": "qn", "ad": "qa"}), on="_o", how="left").join(TXT.rename({"_id": "_sib", "nm": "sn", "ad": "sa"}), on="_sib", how="left")
    log(f"{split}: prototype comparisons {ex.height:,}")
    w = 12
    ex = ex.with_columns(pl.Series("pn", process.cpdist(ex["qn"].to_list(), ex["sn"].to_list(), scorer=fuzz.ratio, workers=w, dtype=np.uint8)),
                         pl.Series("pa", process.cpdist(ex["qa"].to_list(), ex["sa"].to_list(), scorer=fuzz.token_set_ratio, workers=w, dtype=np.uint8)),
                         ((pl.col("qa") == pl.col("sa")) & (pl.col("qa") != "")).alias("pa_eq"))
    pr = ex.group_by(K).agg(pl.col("pn").max().alias("proto_name"), pl.col("pa").max().alias("proto_addr"),
                            pl.col("pn").mean().alias("proto_name_mean"), pl.col("pa_eq").any().cast(pl.Int8).alias("proto_addr_eq"))
    del ex
    d = d.join(pr, on=K, how="left")
    s1t = d.select("s1_id", "cand_id", "_o").join(TXT.rename({"_id": "_o", "nm": "qn", "ad": "qa"}), on="_o", how="left") \
        .join(TXT.rename({"_id": "s1_id", "nm": "sn", "ad": "sa"}), on="s1_id", how="left")
    d = d.with_columns(pl.Series("s1_name", process.cpdist(s1t["qn"].to_list(), s1t["sn"].to_list(), scorer=fuzz.ratio, workers=w, dtype=np.uint8)),
                       pl.Series("s1_addr", process.cpdist(s1t["qa"].to_list(), s1t["sa"].to_list(), scorer=fuzz.token_set_ratio, workers=w, dtype=np.uint8)))
    del s1t
    # competitor = best other owner of the record
    top2 = d.filter(pl.col("r_rank") <= 2).select("cand_id", "r_rank", "s1_id", "prob", "sib_n", "sib_same", "proto_name", "proto_addr", "s1_name", "s1_addr", "lp")
    c1 = top2.filter(pl.col("r_rank") == 1).drop("r_rank"); c2 = top2.filter(pl.col("r_rank") == 2).drop("r_rank")
    cc = [c for c in c1.columns if c != "cand_id"]
    d = d.join(c1.rename({c: "c1_" + c for c in cc}), on="cand_id", how="left").join(c2.rename({c: "c2_" + c for c in cc}), on="cand_id", how="left")
    isb = pl.col("r_rank") == 1
    for c in cc:
        d = d.with_columns(pl.when(isb).then(pl.col("c2_" + c)).otherwise(pl.col("c1_" + c)).alias("comp_" + c))
    d = d.drop([x for x in d.columns if x.startswith("c1_") or x.startswith("c2_")] + ["comp_s1_id"])
    for c in ("sib_n", "proto_name", "proto_addr", "s1_name", "s1_addr", "lp"):
        d = d.with_columns((pl.col(c).cast(pl.Float32) - pl.col("comp_" + c).cast(pl.Float32)).alias("d_" + c))
    d = d.join(gt.with_columns(pl.lit(1).cast(pl.Int8).alias("y")).rename({"cand_id": "_o"}), on=["s1_id", "_o"], how="left") \
        .with_columns(pl.col("y").fill_null(0), (pl.col("_o").hash(11) % 2).cast(pl.Int8).alias("rf"))
    log(f"{split}: features ready ({time.time()-t:.0f}s), {d.width} cols")
    return d

d1 = features("1x"); d1.write_parquet(f"{O}/feat_1x.parquet")
FEATS = [c for c in d1.columns if c not in ("s1_id", "cand_id", "_o", "y", "rf")]
log(f"features: {FEATS}")
P = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=7)
PR = dict(P, objective="lambdarank", lambdarank_truncation_level=3, eval_at=[1])

def fit_rank(tr):
    g = tr.filter(pl.col("r_n") >= 2).sort("cand_id")
    g = g.filter(pl.col("y").sum().over("cand_id") >= 1)
    sizes = g.group_by("cand_id", maintain_order=True).len()["len"].to_numpy()
    ds = lgb.Dataset(g.select(FEATS).to_numpy().astype(np.float32), g["y"].to_numpy(), group=sizes, feature_name=FEATS)
    return lgb.train(PR, ds, 400)

def fit_bin(tr, extra):
    F = FEATS + extra
    es = (tr["s1_id"].hash(5) % 20 == 0).to_numpy()
    X = tr.select(F).to_numpy().astype(np.float32); y = tr["y"].to_numpy()
    dt = lgb.Dataset(X[~es], y[~es], feature_name=F); dv = lgb.Dataset(X[es], y[es], reference=dt)
    return lgb.train(P, dt, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])

models = {}
for f in (0, 1):
    tr = d1.filter(pl.col("rf") == f)
    rk = fit_rank(tr); models[("rank", f)] = rk
    log(f"ranker fold {f}: {rk.num_trees()} trees")
# OOF ranker score as a feature (record-fold f scored by the model of fold 1-f)
def add_rank(d):
    s = np.zeros(d.height, np.float32); rf = d["rf"].to_numpy(); X = d.select(FEATS).to_numpy().astype(np.float32)
    for f in (0, 1):
        m = rf == f; s[m] = models[("rank", 1 - f)].predict(X[m], num_threads=a.threads)
    return d.with_columns(pl.Series("rk", s)).with_columns(pl.col("rk").rank("ordinal", descending=True).over("cand_id").alias("rk_rank"),
                                                           (pl.col("rk") - pl.col("rk").max().over("cand_id")).alias("rk_gap"))
d1 = add_rank(d1)
EXTRA = ["rk", "rk_rank", "rk_gap"]
for f in (0, 1):
    models[("bin", f)] = fit_bin(d1.filter(pl.col("rf") == f), EXTRA)
    log(f"binary fold {f}: {models[('bin', f)].best_iteration} iters")
imp = sorted(zip(FEATS + EXTRA, models[("bin", 0)].feature_importance("gain")), key=lambda x: -x[1])[:25]
log("top gain: " + ", ".join(f"{n}={v/1e3:.0f}k" for n, v in imp))

def add_bin(d):
    q = np.zeros(d.height, np.float32); rf = d["rf"].to_numpy(); X = d.select(FEATS + EXTRA).to_numpy().astype(np.float32)
    for f in (0, 1):
        m = rf == f; b = models[("bin", 1 - f)]; q[m] = b.predict(X[m], num_iteration=b.best_iteration, num_threads=a.threads)
    return d.with_columns(pl.Series("q", q))
d1 = add_bin(d1)

def owner_eval(d, tag):
    g = d.filter((pl.col("r_n") >= 2) & (pl.col("y").sum().over("cand_id") >= 1))
    base = g.filter(pl.col("r_rank") == 1).select("cand_id", pl.col("y").alias("yb"))
    rkc = g.filter(pl.col("rk_rank") == 1).select("cand_id", pl.col("y").alias("yr"))
    qc = g.sort("q", descending=True).group_by("cand_id").first().select("cand_id", pl.col("y").alias("yq"))
    e = base.join(rkc, on="cand_id").join(qc, on="cand_id")
    amb = g.select("cand_id", "addr_empty", "name_owners").unique("cand_id")
    e = e.join(amb, on="cand_id")
    for name, sub in (("all", e), ("ambiguous", e.filter(pl.col("addr_empty") & (pl.col("name_owners") >= 2))),
                      ("non-amb", e.filter(~(pl.col("addr_empty") & (pl.col("name_owners") >= 2))))):
        log(f"{tag} owner choice [{name}] groups {sub.height:,}: champion correct {sub['yb'].mean():.4f}; ranker {sub['yr'].mean():.4f} "
            f"(fixed {int(((sub['yr']==1)&(sub['yb']==0)).sum()):,}, broke {int(((sub['yr']==0)&(sub['yb']==1)).sum()):,}); "
            f"binary-q {sub['yq'].mean():.4f} (fixed {int(((sub['yq']==1)&(sub['yb']==0)).sum()):,}, broke {int(((sub['yq']==0)&(sub['yb']==1)).sum()):,})")
owner_eval(d1, "1x")

def decide_eval(d, full_path, tag, rule=None):
    full = pl.read_parquet(full_path).select(*K, "prob")
    new = full.join(d.select(*K, "q"), on=K, how="left").with_columns(pl.coalesce("q", "prob").alias("prob")).drop("q")
    res = {}
    base = macro_f05(keep_by_rank(assign_owner(full), .60, .75).select(K), gt, S1_IDS); res["champion@.60/.75"] = base
    if rule is None:
        best = (None, -1)
        owned = assign_owner(new)
        for tt in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            for trr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
                if tt > trr: continue
                m = macro_f05(keep_by_rank(owned, tt, trr).select(K), gt, S1_IDS)
                if m["macro_f05"] > best[1]: best = ((tt, trr), m["macro_f05"])
        rule = best[0]
    m = macro_f05(keep_by_rank(assign_owner(new), *rule).select(K), gt, S1_IDS); res[f"meta@{rule}"] = m
    for k, v in res.items():
        log(f"{tag} {k}: F0.5 {v['macro_f05']:.5f} P {v['micro_precision']:.5f} R {v['micro_recall']:.5f} singleton {v['singleton_acc']:.5f}")
    return rule, res
rule, r1 = decide_eval(d1, f"{R}/train_final.parquet", "1x")
import pickle
for k, m in models.items(): m.save_model(f"{O}/{k[0]}_fold{k[1]}.txt")
# ambiguous confirmation: coverage / precision of the chosen owner for ambiguous records by q
amb = d1.filter(pl.col("addr_empty") & (pl.col("name_owners") >= 2) & (pl.col("r_n") >= 2))
top = amb.sort("q", descending=True).group_by("cand_id").first()
for thr in (0.5, 0.6, 0.67, 0.75, 0.9):
    x = top.filter(pl.col("q") >= thr)
    log(f"ambiguous records (>=2 owners) {top.height:,}: chosen q>={thr}: coverage {x.height/ max(top.height,1):.4f}, precision {x['y'].mean() if x.height else float('nan'):.4f}")
d1.select(*K, "q", "rk", "y").write_parquet(f"{O}/oof_1x.parquet")
del d1
d2 = features("2x")
d2 = add_rank(d2); d2 = add_bin(d2)
owner_eval(d2, "2x")
_, r2 = decide_eval(d2, f"{R}/x2_final.parquet", "2x", rule=rule)
d2.select(*K, "q", "rk", "y").write_parquet(f"{O}/oof_2x.parquet")
json.dump({"rule": rule, "1x": {k: v for k, v in r1.items()}, "2x": {k: v for k, v in r2.items()}}, open(f"{O}/metrics.json", "w"), indent=1, default=float)
# test: both fold models averaged
PREP["test"] = prep("test")
d3 = features("test")
X = d3.select(FEATS).to_numpy().astype(np.float32)
d3 = d3.with_columns(pl.Series("rk", (models[("rank", 0)].predict(X, num_threads=a.threads) + models[("rank", 1)].predict(X, num_threads=a.threads)) / 2))
d3 = d3.with_columns(pl.col("rk").rank("ordinal", descending=True).over("cand_id").alias("rk_rank"), (pl.col("rk") - pl.col("rk").max().over("cand_id")).alias("rk_gap"))
X = d3.select(FEATS + EXTRA).to_numpy().astype(np.float32)
q = sum(models[("bin", f)].predict(X, num_iteration=models[("bin", f)].best_iteration, num_threads=a.threads) for f in (0, 1)) / 2
full = pl.read_parquet(f"{R}/test_final_iu.parquet", columns=[*K, "prob"]).join(d3.select(K).with_columns(pl.Series("q", q.astype(np.float32))), on=K, how="left")     .with_columns(pl.coalesce("q", "prob").alias("prob")).drop("q")
full.write_parquet(f"{O}/test_final_iu_meta.parquet"); log(f"test: {d3.height:,} population edges rescored; rule {rule}")
log("DONE")
