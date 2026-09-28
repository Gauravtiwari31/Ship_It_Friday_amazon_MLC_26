"""leader_gap F5 (fr_cascade_bi): as fr_cascade.py + learned-retriever features (f_bi_top1, f_bi_cos1, f_bi_gap) in ALL stages of the
unseen-country chain; stage 1 is retrained with them (it also drives the candidate filter). Original docstring:
leader_gap C2: compact candidate cascade for countries WITHOUT training labels (France), mirroring the production
unseen chain (run_pipeline.unseen_stage2 / unseen_stage3) on the filtered candidates only. Trained on India+US labels only
(no test fitting). candidate generation = TF-IDF blocking -> no-word-risk stage-1 filter (p1 >= tau).
matcher = contexts recomputed over the filtered pairs -> no-word-risk stage 2 (retrained, cross-fitted) -> chain = mean(p1, p2)
-> stage 3 on the chain's zone (0.01<chain<0.99) with the e5-small CE logit + context (retrained) -> decided at 0.95.
Steps: filt, s2, ce (GPU for missing logits), s3. Output: test France final probabilities + comparison with v17r_nodba France."""
import argparse, json, os, sys, time
os.environ.setdefault("POLARS_MAX_THREADS", "12")
import numpy as np, polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
import train as TR
TR.LGB_PARAMS["num_threads"] = 12
from decide import assign_owner, keep_by_rank
from features import WORD_RISK_FEATURES
from config import UNSEEN_EXTRA_EXCLUDE
from stage2 import ce_context, write_context_parts, write_sibling_context_parts, write_source_context_parts
from train import load_part, part_files, score_parts, train_all
ap = argparse.ArgumentParser(); ap.add_argument("--steps", default="bi,s1,filt,s2,ce,s3"); ap.add_argument("--tau", type=float, default=0.003)
ap.add_argument("--out", default="work/audit/leader_gap/lg_frcascade_bi"); ap.add_argument("--work", default="work_v17r"); ap.add_argument("--bicos", default="")
a = ap.parse_args(); steps = set(a.steps.split(","))
T0 = time.time(); K = ["s1_id", "cand_id"]; W, O = a.work, a.out; MD = f"{O}/model"; os.makedirs(MD, exist_ok=True)
WV = "work_v17r"  # cross-encoder fold models / cached logits (pair-keyed)
EXCL = set(WORD_RISK_FEATURES) | set(UNSEEN_EXTRA_EXCLUDE)
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:5.0f}s] {m}", flush=True)
BASE0 = {s: [f"{W}/{s}_feats", f"{W}/{s}_extra", f"{W}/{s}_numrel"] for s in ("train", "test")}
BASE = {s: BASE0[s] + [f"{O}/{s}_bi"] for s in ("train", "test")}
P1 = {s: f"{O}/{s}_p1new" for s in ("train", "test")}
ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
def sizes(d): return [pl.scan_parquet(f).select(pl.len()).collect().item() for f in part_files(d)]
if "bi" in steps:  # retriever features, row-aligned with the base parts
    for split in ("train", "test"):
        B = pl.read_parquet(f"work/audit/leader_gap/bienc_cands_{split}.parquet").select(*K, pl.lit(1.0).cast(pl.Float32).alias("f_bi_top1"),
            pl.col("bi_cos").cast(pl.Float32).alias("f_bi_cos1"), (pl.col("bi_cos") - pl.col("bi_cos2")).cast(pl.Float32).alias("f_bi_gap"))
        os.makedirs(f"{O}/{split}_bi", exist_ok=True)
        for i, f in enumerate(part_files(BASE0[split][0])):
            pl.read_parquet(f, columns=K).join(B, on=K, how="left", maintain_order="left").with_columns(pl.col("f_bi_top1").fill_null(0.0))                 .drop(K).write_parquet(f"{O}/{split}_bi/part_{i:03d}.parquet")
        log(f"{split}: retriever features written")
if "s1" in steps:  # no-word-risk stage 1 with the retriever features (cross-fitted OOF on train, mean of folds on test)
    m1 = train_all(BASE["train"], MD, prefix="ustage1", log=log, exclude=EXCL)
    for split, oof in (("train", True), ("test", False)):
        sc = score_parts(BASE[split], m1, oof=oof, log=lambda m: None)
        sz = sizes(BASE0[split][0]); offs = np.concatenate([[0], np.cumsum(sz)]); os.makedirs(P1[split], exist_ok=True)
        for j in range(len(sz)):
            sc.slice(int(offs[j]), int(sz[j])).select(pl.col("prob").alias("p1_prob")).write_parquet(f"{P1[split]}/part_{j:03d}.parquet")
        log(f"{split}: new stage-1 probabilities written")
FB = {s: f"{O}/{s}_base" for s in BASE}; CTX = {s: [f"{O}/{s}_p1ctx", f"{O}/{s}_srcctx", f"{O}/{s}_sibctx"] for s in BASE}
DIRS = {s: [FB[s]] + CTX[s] for s in BASE}
tn = pl.read_parquet(f"{W}/test_norm.parquet", columns=["entity_id", "src", "country"])
tr_c = set(pl.read_parquet(f"{W}/train_norm.parquet", columns=["src", "country"]).filter(pl.col("src") == 1)["country"].unique().to_list())
UNSEEN = tn.filter((pl.col("src") == 1) & ~pl.col("country").is_in(list(tr_c)))["entity_id"]
if "filt" in steps:
    for split in ("train", "test"):
        os.makedirs(FB[split], exist_ok=True); scs = []; n0 = n1 = 0
        for i in range(len(part_files(BASE[split][0]))):
            df = load_part(i, BASE[split]); p1 = pl.read_parquet(part_files(P1[split])[i], columns=["p1_prob"])["p1_prob"]
            assert len(p1) == df.height
            keep = p1 >= a.tau
            if split == "test": keep = keep & df["s1_id"].is_in(UNSEEN.implode())
            n0 += df.height; df = df.filter(keep); n1 += df.height
            df.write_parquet(f"{FB[split]}/part_{i:03d}.parquet"); scs.append(df.select(*K).with_columns(p1.filter(keep).alias("prob")))
        sc = pl.concat(scs); sz = sizes(FB[split]); log(f"{split}: kept {n1:,} of {n0:,} pairs")
        norm = pl.read_parquet(f"{W}/{split}_norm.parquet", columns=["entity_id", "src", "addr_tokens", "addr_nums", "name_core"])
        write_context_parts(sc, sz, CTX[split][0])
        write_source_context_parts(sc, norm.select("entity_id", "src"), sz, CTX[split][1])
        write_sibling_context_parts(sc, norm.filter(pl.col("src") != 1).select("entity_id", "addr_tokens", "addr_nums", "name_core"), sz, CTX[split][2])
        log(f"{split}: contexts written"); del sc, scs, norm
if "s2" in steps:
    m2 = train_all(DIRS["train"], MD, prefix="ustage2", log=log, exclude=EXCL)
    for split, oof in (("train", True), ("test", False)):
        s2 = score_parts(DIRS[split], m2, oof=oof, log=lambda m: None).select(*K, pl.col("prob").alias("p2"))
        p1 = pl.concat([pl.read_parquet(f, columns=["p1_prob"]) for f in part_files(CTX[split][0])])["p1_prob"]
        s2.with_columns(p1.alias("p1")).with_columns(((pl.col("p1") + pl.col("p2")) / 2).alias("prob")).write_parquet(f"{O}/{split}_chain.parquet")
    log("stage 2 done")
def zone(split): return pl.read_parquet(f"{O}/{split}_chain.parquet").filter((pl.col("prob") > .01) & (pl.col("prob") < .99))
def have(sp):
    fs = [f"{WV}/ce_model_v9/{sp}_scores.parquet"] + ([f"{WV}/ce_model_v9/train_scores_unseen.parquet"] if sp == "train" else []) \
        + [f for f in (f"{O}/ce_{sp}_new.parquet", f"work/audit/leader_gap/frcascade_t003/ce_{sp}_new.parquet") if os.path.exists(f)]
    return pl.concat([pl.read_parquet(f, columns=[*K, "ce_logit"]).with_columns(pl.col("ce_logit").cast(pl.Float32)) for f in fs]).unique(K)
if "ce" in steps:
    import cross_encoder
    folds = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]) for f in part_files(f"{W}/train_feats")]).unique("s1_id")
    for sp in ("train", "test"):
        z = zone(sp).select(K); todo = z.join(have(sp), on=K, how="anti")
        if sp == "train": todo = todo.join(folds, on="s1_id")
        log(f"{sp}: zone {z.height:,}, without logit {todo.height:,}")
        if todo.height:
            cross_encoder.BACKBONE = "intfloat/multilingual-e5-small"
            cross_encoder.score_pairs("dataset", sp, f"{WV}/ce_model_v9", todo, logger=log).with_columns(pl.col("ce_logit").cast(pl.Float32)) \
                .write_parquet(f"{O}/ce_{sp}_new.parquet")
SFX = "_bc" if a.bicos else ""
def write_zone(split):
    z = zone(split).select(*K, pl.col("prob").alias("p2_unseen_base"))
    t = z.join(have("test" if split == "test" else "train"), on=K, how="left")
    log(f"{split}: zone {t.height:,}, missing logits {t['ce_logit'].null_count():,}")
    v = t.filter(pl.col("ce_logit").is_not_null())
    t = t.join(pl.concat([v.select(K), ce_context(v.select(*K, "ce_logit"))], how="horizontal"), on=K, how="left").rename({"ce_logit": "p2_ce_logit"})
    if a.bicos:
        t = t.join(pl.read_parquet(f"{a.bicos}_{'test' if split == 'test' else 'train'}.parquet").select(*K, pl.col("bi_cos").cast(pl.Float32).alias("p2_bi_cos")), on=K, how="left")
    zd = f"{O}/{split}_uzone{SFX}"; os.makedirs(zd, exist_ok=True)
    for i in range(len(part_files(DIRS[split][0]))):
        load_part(i, DIRS[split]).join(t, on=K, how="inner", maintain_order="left").write_parquet(f"{zd}/part_{i:03d}.parquet")
    return zd
if "s3" in steps:
    m3 = train_all([write_zone("train")], MD, prefix="ustage3" + SFX, log=log, exclude=EXCL)
    p3 = score_parts([write_zone("test")], m3, oof=False, log=lambda m: None).select(*K, pl.col("prob").alias("p3"))
    fin = pl.read_parquet(f"{O}/test_chain.parquet").select(*K, "prob").join(p3, on=K, how="left").with_columns(pl.coalesce("p3", "prob").alias("prob")).drop("p3")
    fin.write_parquet(f"{O}/test_final_fr{SFX}.parquet")
    res = {"tau": a.tau, "fr_pairs": fin.height, "fr_cand_per_s1": round(fin.height / len(UNSEEN), 3)}
    ref = []
    if not os.path.exists("submission/v17r_nodba/matching_results.tsv"):  # packaged run: no earlier submission to compare with
        json.dump(res, open(f"{O}/metrics{SFX}.json", "w"), indent=1); log(json.dumps(res)); log("DONE"); sys.exit(0)
    with open("submission/v17r_nodba/matching_results.tsv", encoding="utf-8") as fh:
        fh.readline()
        for l in fh:
            s, ids = l.rstrip("\n").split("\t")
            if ids: ref += [(s, c) for c in ids.split(",")]
    ref = pl.DataFrame(ref, schema=K, orient="row").filter(pl.col("s1_id").is_in(UNSEEN.implode()))
    dba = pl.read_parquet("work/audit/v17r/dba_only_test.parquet").select(K)
    for tag, x, t in (("dba", fin, 0.95), ("dba_flat85", fin, 0.85), ("dba_flat80", fin, 0.80)):
        m = keep_by_rank(assign_owner(x), t, t).select(K); m.write_parquet(f"{O}/fr_matches_{tag}{SFX}.parquet")
        res[tag] = dict(matches=m.height, vs_v17r_nodba_added=m.join(ref, on=K, how="anti").height, removed=ref.join(m, on=K, how="anti").height,
                        s1_changed=pl.concat([m.join(ref, on=K, how="anti"), ref.join(m, on=K, how="anti")])["s1_id"].n_unique())
    json.dump(res, open(f"{O}/metrics{SFX}.json", "w"), indent=1); log(json.dumps(res)); log("DONE")
