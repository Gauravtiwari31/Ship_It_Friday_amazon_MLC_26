"""leader_gap R1: supervised bi-encoder retrieval (S2/S3 record -> S1), training data only, out-of-fold by S1 fold.

Train: GT pairs whose S1 is in --fold (one record per S1 -> no in-batch false negatives), InfoNCE over in-batch
positives + one mined hard negative per query (highest-scoring wrong S1 among the record's current candidates,
champion OOF; random same-country S1 if none). Shared encoder, mean pooling, "query: name | address".
Eval (S1 of the other fold never seen as positives): index = ALL S1 of the country; queries = every current
blocking-FN record owned by an eval-fold S1 + a uniform sample of records not owned by a train-fold S1.
Outputs: model.pt, s1 embeddings, top-k parquet, metrics json/log.
"""
import os, sys, time, math, argparse, json, glob
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="intfloat/multilingual-e5-small")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n-train", type=int, default=600000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max-len", type=int, default=80)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--eval-frac", type=float, default=0.08)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--out", default="work/audit/leader_gap/bienc_small_f0")
ap.add_argument("--skip-train", action="store_true")
ap.add_argument("--train-country", default="")
ap.add_argument("--eval-country", default="")
ap.add_argument("--no-eval", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
T0 = time.time()
LOGF = open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8")
def log(m):
    s = f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:.0f}s] {m}"; print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
K = ["s1_id", "cand_id"]; R = "work/audit/ceiling6h/retrieval"
EF = 1 - a.fold

raw = pl.read_parquet("work_v17r/train_raw.parquet").with_columns(
    ("query: " + pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t"))
T = raw.select("entity_id", "t")
fold = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]) for f in sorted(glob.glob("work_v17r/train_feats/*.parquet"))]).unique("s1_id")
s1all = raw.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1_id"), "country") \
    .join(fold, on="s1_id", how="left").with_columns(pl.col("fold").fill_null((pl.col("s1_id").hash(3) % 2).cast(pl.Int32)))
gt = pl.read_parquet("work_v17r/train_gt_pairs.parquet").select(K).join(s1all.select("s1_id", "fold"), on="s1_id")
log(f"S1 {s1all.height:,}; GT pairs {gt.height:,}; train fold {a.fold}")

tok = AutoTokenizer.from_pretrained(a.backbone)
model = AutoModel.from_pretrained(a.backbone).cuda()

def enc(texts):
    b = tok(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
    b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    e = (h * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(e.float(), dim=-1)

mp = os.path.join(a.out, "model.pt")
if not a.skip_train and not os.path.exists(mp):
    trpool = gt.filter(pl.col("fold") == a.fold)
    if a.train_country:
        trpool = trpool.join(s1all.filter(pl.col("country") == a.train_country).select("s1_id"), on="s1_id", how="semi")
    tr = trpool.sample(fraction=1.0, shuffle=True, seed=1).unique("s1_id", keep="first") \
        .sample(fraction=1.0, shuffle=True, seed=2).head(a.n_train)
    hn = pl.scan_parquet(f"{R}/champion_oof_1x.parquet").join(tr.lazy().select("cand_id", pl.col("s1_id").alias("_t")), on="cand_id") \
        .filter(pl.col("s1_id") != pl.col("_t")).sort("prob", descending=True).group_by("cand_id").agg(pl.col("s1_id").first().alias("neg")).collect()
    tr = tr.join(hn, on="cand_id", how="left").join(raw.select(pl.col("entity_id").alias("cand_id"), "country"), on="cand_id")
    ids_c = {c: g["s1_id"].to_numpy() for (c,), g in s1all.group_by("country")}
    rng = np.random.default_rng(0)
    rnd = [ids_c[c][rng.integers(len(ids_c[c]))] for c in tr["country"].to_list()]
    tr = tr.with_columns(pl.Series("_r", rnd)).with_columns(pl.coalesce("neg", "_r").alias("neg")).drop("_r")
    log(f"train pairs {tr.height:,} (one record per S1); mined hard negatives {hn.height:,}")
    tr = tr.join(T.rename({"entity_id": "cand_id", "t": "tq"}), on="cand_id", how="left", maintain_order="left") \
        .join(T.rename({"entity_id": "s1_id", "t": "tp"}), on="s1_id", how="left", maintain_order="left") \
        .join(T.rename({"entity_id": "neg", "t": "tn"}), on="neg", how="left", maintain_order="left")
    tq, tp, tn = tr["tq"].to_list(), tr["tp"].to_list(), tr["tn"].to_list()
    pid = tr["s1_id"].hash(5).to_numpy().view(np.int64)
    nid = tr["neg"].hash(5).to_numpy().view(np.int64)
    B = a.batch; steps = len(tq) // B
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, int(0.03 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    model.gradient_checkpointing_enable()  # 3 x batch sequences per step: activations would spill out of 16 GB VRAM
    ce = torch.nn.CrossEntropyLoss(); model.train(); t = time.time(); run = 0.0
    for st in range(steps):
        sl = slice(st * B, (st + 1) * B)
        q = enc(tq[sl]); d = enc(tp[sl] + tn[sl])
        s = q @ d.T / a.tau
        ids = torch.as_tensor(np.concatenate([pid[sl], nid[sl]]), device="cuda")
        qid = torch.as_tensor(pid[sl], device="cuda")
        mask = ids[None, :] == qid[:, None]
        ar = torch.arange(len(qid), device="cuda"); mask[ar, ar] = False
        s = s.masked_fill(mask, -1e4)
        loss = ce(s, ar)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if st % 200 == 0 or st == steps - 1:
            log(f"step {st}/{steps} loss {run:.4f} ({(st + 1) * B / (time.time() - t):.0f} q/s)")
    torch.save(model.state_dict(), mp); log("saved model")
    del tq, tp, tn
elif os.path.exists(mp):
    model.load_state_dict(torch.load(mp, map_location="cuda")); log("loaded model")
else:
    log("zero-shot (no training)")
model.eval()
if a.no_eval:
    log("DONE (no eval)"); sys.exit(0)

@torch.no_grad()
def embed(texts, bs=1024):
    order = np.argsort([len(x) for x in texts]); out = np.zeros((len(texts), model.config.hidden_size), np.float16)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]; out[idx] = enc([texts[j] for j in idx]).half().cpu().numpy()
    return out

# ---- eval queries
fn = pl.read_parquet(f"{R}/fn_1x.parquet", columns=["s1_id", "cand_id", "class", "old_ambiguous"])
bfn = fn.filter(pl.col("class") == "blocking_false_negative").join(s1all.select("s1_id", "fold", "country"), on="s1_id")
bfn = bfn.filter(pl.col("country") == a.eval_country) if a.eval_country else bfn.filter(pl.col("fold") == EF)
owner = gt.select(pl.col("cand_id"), pl.col("s1_id").alias("true_s1"), pl.col("fold").alias("tfold"))
recs = raw.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("cand_id"), "country", "src").join(owner, on="cand_id", how="left")
elig = recs.filter(pl.col("country") == a.eval_country) if a.eval_country else recs.filter(pl.col("tfold").is_null() | (pl.col("tfold") == EF))
uni = elig.sample(fraction=a.eval_frac, seed=3)
qset = pl.concat([uni.select("cand_id"), bfn.select("cand_id")]).unique().join(recs, on="cand_id")
log(f"eval: blocking-FN queries {bfn.height:,} (non-amb {int((~bfn['old_ambiguous']).sum()):,}); uniform sample {uni.height:,} "
    f"of {elig.height:,} eligible records; total queries {qset.height:,}")
topk_path = os.path.join(a.out, "topk.parquet")
res = []
for (c,), s1c in s1all.group_by("country"):
    if a.eval_country and c != a.eval_country:
        continue
    qc = qset.filter(pl.col("country") == c)
    S = embed(s1c.join(T.rename({"entity_id": "s1_id"}), on="s1_id", how="left", maintain_order="left")["t"].to_list())
    np.save(os.path.join(a.out, f"s1emb_{c}.npy"), S)
    Q = embed(qc.join(T.rename({"entity_id": "cand_id"}), on="cand_id", how="left", maintain_order="left")["t"].to_list())
    log(f"{c}: embedded {len(S):,} S1, {len(Q):,} queries")
    St = torch.from_numpy(S).cuda()
    ids = s1c["s1_id"].to_numpy(); qids = qc["cand_id"].to_numpy()
    for i in range(0, len(Q), 256):
        sim = torch.from_numpy(Q[i:i + 256]).cuda() @ St.T
        v, ix = sim.topk(a.k, dim=1); v = v.float()
        v, ix = v.cpu().numpy(), ix.cpu().numpy()
        n = len(ix)
        res.append(pl.DataFrame({"cand_id": np.repeat(qids[i:i + n], a.k), "s1_id": ids[ix.ravel()],
                                 "rank": np.tile(np.arange(1, a.k + 1, dtype=np.int16), n), "cos": v.ravel().astype(np.float32)}))
    del St; torch.cuda.empty_cache()
top = pl.concat(res); top.write_parquet(topk_path); log(f"top-{a.k}: {top.height:,} rows")

# ---- metrics
cur = pl.scan_parquet(f"{R}/champion_oof_1x.parquet").select(K).join(top.lazy().select("cand_id").unique(), on="cand_id").collect() \
    .with_columns(pl.lit(True).alias("in_cur"))
top = top.join(cur, on=K, how="left").with_columns(pl.col("in_cur").fill_null(False)) \
    .join(gt.select(*K, pl.lit(True).alias("y")), on=K, how="left").with_columns(pl.col("y").fill_null(False))
M = {}
def rec_at(df_true, name):
    d = df_true.join(top.select("cand_id", "s1_id", "rank"), on=["cand_id", "s1_id"], how="left")
    r = {f"R@{k}": round(float((d["rank"].fill_null(99) <= k).mean()), 4) for k in (1, 3, 5, 10, 20)}
    r["n"] = d.height; M[name] = r; log(f"{name:34s} {r}")
own_uni = uni.filter(pl.col("true_s1").is_not_null()).select("cand_id", pl.col("true_s1").alias("s1_id"), "country", "src")
rec_at(own_uni, "all matched (uniform, eval fold)")
for c in ("India", "US"): rec_at(own_uni.filter(pl.col("country") == c), f"  {c}")
for s in (2, 3): rec_at(own_uni.filter(pl.col("src") == s), f"  S{s}")
rec_at(bfn.select(K), "blocking FN (all)")
rec_at(bfn.filter(~pl.col("old_ambiguous")).select(K), "blocking FN NON-AMBIGUOUS")
nb = bfn.filter(~pl.col("old_ambiguous")).join(recs.select("cand_id", "country", "src"), on="cand_id")
for c in ("India", "US"): rec_at(nb.filter(pl.col("country") == c).select(K), f"  non-amb {c}")
for s in (2, 3): rec_at(nb.filter(pl.col("src") == s).select(K), f"  non-amb S{s}")
rec_at(bfn.filter(pl.col("old_ambiguous")).select(K), "blocking FN ambiguous")
# economics: NEW pairs (not current candidates), per uniform-sample query, and recovered blocking FN
u = top.join(uni.select("cand_id"), on="cand_id", how="semi")
nbk = bfn.filter(~pl.col("old_ambiguous")).select(K)
N_REC = recs.filter(pl.col("country") == a.eval_country).height if a.eval_country else 10_320_219
N_NB_1X = int(fn.filter((pl.col("class") == "blocking_false_negative") & ~pl.col("old_ambiguous")).join(s1all.filter(pl.col("country") == a.eval_country).select("s1_id"), on="s1_id", how="semi").height) if a.eval_country else 75_376
econ = []
for k in (1, 3, 5, 10, 20):
    for cmin in (0.0, 0.8, 0.85, 0.9):
        x = u.filter((pl.col("rank") <= k) & (pl.col("cos") >= cmin) & ~pl.col("in_cur"))
        new_per_q = x.height / uni.height; false_per_q = int((~x["y"]).sum()) / uni.height
        rec_nb = nbk.join(top.filter((pl.col("rank") <= k) & (pl.col("cos") >= cmin)), on=K, how="semi").height / max(nbk.height, 1)
        full_new = new_per_q * N_REC; full_true_nb = rec_nb * N_NB_1X
        econ.append(dict(k=k, cos_min=cmin, new_pairs_per_query=round(new_per_q, 3), est_full_new_pairs=int(full_new),
                         nonamb_bfn_recall=round(rec_nb, 4), est_full_nonamb_recovered=int(full_true_nb),
                         est_false_per_true=round(false_per_q * N_REC / max(full_true_nb, 1), 1)))
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
E = pl.DataFrame(econ); print(E); LOGF.write(str(E) + "\n")
M["econ"] = econ
json.dump(M, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
log("DONE")
