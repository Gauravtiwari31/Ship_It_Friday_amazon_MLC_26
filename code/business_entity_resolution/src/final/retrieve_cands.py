"""leader_gap R1b: record-centric learned retrieval candidates (top-1 S1 per S2/S3 record) for train and test.
Two cross-fitted e5-small bi-encoders: A (trained on GT pairs of fold-0 S1), B (fold-1 S1).
Train (out-of-fold): a record owned by a fold-0 S1 is embedded / retrieved with B, owned by fold-1 with A (the model never
saw that record's pair); unmatched records by hash. Index = all S1 of the record's country, embedded by the same model.
Test: every record with model A (no test label or pseudo-label is used; frozen models; all countries alike).
Output: work/audit/leader_gap/bienc_cands_{train,test}.parquet: s1_id, cand_id, bi_cos (top-1), bi_cos2 (runner-up)."""
import os, sys, time, glob, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("POLARS_MAX_THREADS", "8")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
ap = argparse.ArgumentParser(); ap.add_argument("--splits", default="train,test"); ap.add_argument("--max-len", type=int, default=80)
a = ap.parse_args()
BB = "intfloat/multilingual-e5-small"; D = "work/audit/leader_gap"; K = ["s1_id", "cand_id"]; T0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:.0f}s] {m}", flush=True)
tok = AutoTokenizer.from_pretrained(BB)
models = {}
for tag, d in (("A", f"{D}/bienc_small_f0"), ("B", f"{D}/bienc_small_f1")):
    m = AutoModel.from_pretrained(BB).cuda(); m.load_state_dict(torch.load(f"{d}/model.pt", map_location="cuda")); m.eval(); models[tag] = m
@torch.no_grad()
def embed(model, texts, bs=1024):
    order = np.argsort([len(x) for x in texts]); out = np.zeros((len(texts), 384), np.float16)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**b).last_hidden_state
        mk = b["attention_mask"].unsqueeze(-1).to(h.dtype); e = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
        out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).half().cpu().numpy()
    return out
@torch.no_grad()
def top2(Q, S):
    St = torch.from_numpy(S).cuda(); v_all, i_all = [], []
    for i in range(0, len(Q), 512):
        v, ix = (torch.from_numpy(Q[i:i + 512]).cuda() @ St.T).topk(2, dim=1)
        v_all.append(v.float().cpu().numpy()); i_all.append(ix.cpu().numpy())
    del St; torch.cuda.empty_cache()
    return np.concatenate(v_all), np.concatenate(i_all)
for split in a.splits.split(","):
    raw = pl.read_parquet(f"work_v17r/{split}_raw.parquet", columns=["entity_id", "business_name", "business_address", "country", "src"])
    raw = raw.with_columns(("query: " + pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t"))
    if split == "train":
        fold = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]) for f in sorted(glob.glob("work_v17r/train_feats/*.parquet"))]).unique("s1_id")
        gt = pl.read_parquet("work_v17r/train_gt_pairs.parquet").select(K).join(fold, on="s1_id", how="left")
        recs = raw.filter(pl.col("src") != 1).join(gt.select(pl.col("cand_id").alias("entity_id"), "fold"), on="entity_id", how="left") \
            .with_columns(pl.when(pl.col("fold") == 0).then(pl.lit("B")).when(pl.col("fold") == 1).then(pl.lit("A"))
                          .otherwise(pl.when(pl.col("entity_id").hash(9) % 2 == 0).then(pl.lit("A")).otherwise(pl.lit("B"))).alias("m"))
    else:
        recs = raw.filter(pl.col("src") != 1).with_columns(pl.lit("A").alias("m"))
    out = []
    for (c,), s1c in raw.filter(pl.col("src") == 1).group_by("country"):
        sid = s1c["entity_id"].to_numpy(); st = s1c["t"].to_list()
        for tag in sorted(recs.filter(pl.col("country") == c)["m"].unique().to_list()):
            q = recs.filter((pl.col("country") == c) & (pl.col("m") == tag))
            if q.height == 0: continue
            S = embed(models[tag], st); Q = embed(models[tag], q["t"].to_list())
            v, ix = top2(Q, S)
            out.append(pl.DataFrame({"s1_id": sid[ix[:, 0]], "cand_id": q["entity_id"].to_numpy(), "bi_cos": v[:, 0], "bi_cos2": v[:, 1]}))
            log(f"{split} {c} model {tag}: {len(S):,} S1, {q.height:,} records")
    res = pl.concat(out); res.write_parquet(f"{D}/bienc_cands_{split}.parquet")
    if split == "train":
        cur = pl.read_parquet("work_v17r/train_cands.parquet", columns=K).with_columns(pl.lit(True).alias("in_cur"))
        r = res.join(cur, on=K, how="left").join(gt.select(*K, pl.lit(True).alias("y")), on=K, how="left").fill_null(False)
        new = r.filter(~pl.col("in_cur"))
        log(f"train: {res.height:,} top-1 pairs; true {int(r['y'].sum()):,}; NEW (not blocking candidates) {new.height:,}, of which true {int(new['y'].sum()):,}")
    log(f"{split}: wrote {res.height:,}")
log("DONE")
