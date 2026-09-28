"""Cross-encoder for the uncertain pairs (v10 candidate).

    python -u src/cross_encoder.py --data ../../dataset --work ../../work_v7 --model model_v9

The gradient-boosting models see about 100 hand-built similarity features. A transformer that reads the
two raw records together ("name | address" of the S1 record and of the candidate) can learn the
generator's own noise patterns (typos, abbreviations, dropped / added words, transliterated scripts)
directly from the strings. It is only needed where the stage-2 model is unsure, so it is trained and
applied on the pairs with 0.01 < p < 0.99 (about 4% of the pairs, but nearly all of the errors).

Backbone: intfloat/multilingual-e5-small (MIT licence, 118M parameters, about 100 languages including
Hindi, the other Indian scripts and French), fine-tuned as a pair classifier (mean-pooled encoder
output -> linear -> logit), bf16 on the GPU.

Cross-fitting follows the S1 folds of the pipeline: the model trained on the fold-k pairs scores the
fold-(1-k) pairs (out-of-fold) and the test pairs; test scores are the mean of the two models.
Outputs: <work>/ce/train_scores.parquet, <work>/ce/test_scores.parquet (s1_id, cand_id, ce_logit).
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKBONE = "intfloat/multilingual-e5-small"
T0 = time.time()


LOG = None
EXTERNAL_LOG = None  # set by run(): the pipeline's logger
VIEW = "raw"         # "raw": the source records as given; "norm": the pipeline's normalised fields (second view)
NORM_WORK = None     # work folder with <split>_norm.parquet (view "norm")


def log(msg):
    """Print a time-stamped progress line (also to <out>/log.txt), or pass it to the pipeline's logger."""
    if EXTERNAL_LOG:
        EXTERNAL_LOG(msg)
        return
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def zone_pairs(work, model, lo, country=None):
    """Pairs with lo < stage-2 probability < 1 - lo: train (with label and fold) and test.
    country: keep only the train pairs of S1 records of this country (per-country models for the
    leave-one-country-out check)."""
    z = (pl.col("prob") > lo) & (pl.col("prob") < 1 - lo)
    tr = pl.scan_parquet(os.path.join(work, f"train_oof_{model}.parquet")).select("s1_id", "cand_id", "prob").filter(z)
    folds = pl.scan_parquet(sorted(glob.glob(os.path.join(work, "train_feats", "*.parquet")))) \
        .select("s1_id", "fold").unique("s1_id")
    gt = pl.scan_parquet(os.path.join(work, "train_gt_pairs.parquet")).select("s1_id", "cand_id", pl.lit(1).alias("label"))
    tr = tr.join(folds, on="s1_id").join(gt, on=["s1_id", "cand_id"], how="left") \
        .with_columns(pl.col("label").fill_null(0))
    if country:
        s1c = pl.scan_parquet(os.path.join(work, "train_norm.parquet")).filter(pl.col("country") == country)             .select(pl.col("entity_id").alias("s1_id"))
        tr = tr.join(s1c, on="s1_id", how="semi")
    tr = tr.collect()
    te = pl.scan_parquet(os.path.join(work, f"test_scored_{model}.parquet")).select("s1_id", "cand_id", "prob").filter(z).collect()
    return tr, te


def texts(data, split, ids):
    """entity_id -> "name | address" for the given ids, from the raw source files (VIEW "raw") or from the
    normalised records (VIEW "norm": transliterated name, expanded address tokens and canonical state)."""
    if VIEW == "norm":
        d = pl.scan_parquet(os.path.join(NORM_WORK, f"{split}_norm.parquet")) \
            .select("entity_id", "name_full", "addr_tokens", "addr_state") \
            .filter(pl.col("entity_id").is_in(ids.implode())).collect() \
            .with_columns((pl.col("name_full").fill_null("") + " | " + pl.col("addr_tokens").fill_null("") + " "
                           + pl.col("addr_state").fill_null("")).str.strip_chars().alias("text"))
        return dict(zip(d["entity_id"].to_list(), d["text"].to_list()))
    rd = dict(separator="\t", quote_char=None, infer_schema=False)
    parts = []
    for s in (1, 2, 3):
        parts.append(pl.scan_csv(os.path.join(data, split, f"{split}_source{s}.tsv"), **rd)
                     .select("entity_id", "business_name", "business_address")
                     .filter(pl.col("entity_id").is_in(ids.implode())).collect())
    d = pl.concat(parts).with_columns(
        (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("text"))
    return dict(zip(d["entity_id"].to_list(), d["text"].to_list()))


class PairModel(torch.nn.Module):
    """Encoder + mean pooling + linear head -> one logit per pair."""

    def __init__(self, backbone):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(backbone)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head(((h * m).sum(1) / m.sum(1)).float()).squeeze(-1)


def batches(a, b, tok, bs, max_len, order):
    """Tokenised batches (dynamic padding) in the given order."""
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation="longest_first",
                  max_length=max_len, return_tensors="pt")
        yield idx, enc["input_ids"].cuda(non_blocking=True), enc["attention_mask"].cuda(non_blocking=True)


def predict(model, tok, a, b, bs, max_len):
    """Logits for all pairs (length-sorted batches for speed)."""
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.zeros(len(a), dtype=np.float32)
    done, t = 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, ids, mask in batches(a, b, tok, bs, max_len, order):
            out[idx] = model(ids, mask).float().cpu().numpy()
            done += len(idx)
            if done % (bs * 2000) < bs:
                log(f"    scored {done:,}/{len(a):,} ({done / (time.time() - t):.0f} pairs/s)")
    return out


def train(tok, a, b, y, args, seed):
    """Fine-tune one pair model for args.epochs epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PairModel(args.backbone).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(a) / args.batch)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    lossf = torch.nn.BCEWithLogitsLoss()
    yt = torch.tensor(y, dtype=torch.float32)
    step, t, run = 0, time.time(), 0.0
    for ep in range(args.epochs):
        model.train()
        # shuffle in length buckets: random order, then sort inside chunks of 100 batches (less padding)
        order = rng.permutation(len(a))
        chunk = args.batch * 100
        lens = np.array([len(a[j]) + len(b[j]) for j in order])
        order = np.concatenate([order[i:i + chunk][np.argsort(lens[i:i + chunk])] for i in range(0, len(order), chunk)])
        bl = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        rng.shuffle(bl)
        order = np.concatenate(bl)
        for idx, ids, mask in batches(a, b, tok, args.batch, args.max_len, order):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(ids, mask)
            loss = lossf(logit, yt[torch.as_tensor(idx)].cuda(non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0:
                rate = step * args.batch / (time.time() - t)
                log(f"    epoch {ep + 1} step {step:,}/{steps:,} loss {run:.4f} "
                    f"({rate:.0f} pairs/s, {(steps - step) * args.batch / rate / 60:.0f} min left)")
    return model


def score_pairs(data, split, model_dir, pairs, max_len=160, logger=None, view="raw", work=None):
    """Cross-encoder logits for more pairs with the two saved fold models of `model_dir`: train pairs
    (s1_id, cand_id, fold) of fold k by the model trained on fold 1 - k (out-of-fold), test pairs by the
    mean of both models. Returns (s1_id, cand_id, ce_logit) in the order of `pairs`."""
    global EXTERNAL_LOG, VIEW, NORM_WORK
    EXTERNAL_LOG = logger or EXTERNAL_LOG
    VIEW, NORM_WORK = view, work
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tx = texts(data, split, pl.concat([pairs["s1_id"], pairs["cand_id"]]).unique())
    a, b = [tx[i] for i in pairs["s1_id"]], [tx[i] for i in pairs["cand_id"]]
    del tx
    out = np.zeros(pairs.height, dtype=np.float32)
    for k in (0, 1):
        m = PairModel(BACKBONE).cuda()
        m.load_state_dict(torch.load(os.path.join(model_dir, f"fold{k}.pt"), map_location="cuda"))
        if split == "train":
            sel = np.flatnonzero(pairs["fold"].to_numpy() == 1 - k)
            log(f"  cross-encoder fold-{k} model: {len(sel):,} out-of-fold train pairs")
            out[sel] = predict(m, tok, [a[j] for j in sel], [b[j] for j in sel], 256, max_len)
        else:
            log(f"  cross-encoder fold-{k} model: {len(a):,} test pairs")
            out += predict(m, tok, a, b, 256, max_len) / 2
        del m
        torch.cuda.empty_cache()
    return pairs.select("s1_id", "cand_id").with_columns(pl.Series("ce_logit", out))


def run(data, work, model, out, zone=0.01, backbone=BACKBONE, epochs=1, batch=64, lr=5e-5, max_len=160,
        limit=0, country=None, no_test=False, logger=None, view="raw"):
    """Cross-fitted cross-encoder scores for the uncertain train and test pairs -> <out>/*_scores.parquet."""
    global LOG, EXTERNAL_LOG, VIEW, NORM_WORK
    os.makedirs(out, exist_ok=True)
    LOG, EXTERNAL_LOG = os.path.join(out, "log.txt"), logger
    VIEW, NORM_WORK = view, work
    args = argparse.Namespace(data=data, work=work, model=model, zone=zone, backbone=backbone, epochs=epochs,
                              batch=batch, lr=lr, max_len=max_len, limit=limit, country=country, no_test=no_test)
    from transformers import AutoTokenizer
    _run(args, AutoTokenizer.from_pretrained(backbone), out)


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="model_v9")
    ap.add_argument("--zone", type=float, default=0.01)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many train / test pairs")
    ap.add_argument("--out", default="ce")
    ap.add_argument("--country", default=None, help="train on the pairs of this country only")
    ap.add_argument("--no-test", action="store_true", help="do not score the test pairs")
    ap.add_argument("--view", default="raw", choices=["raw", "norm"], help="raw records or normalised fields")
    a = ap.parse_args()
    run(a.data, a.work, a.model, os.path.join(a.work, a.out), a.zone, a.backbone, a.epochs, a.batch, a.lr,
        a.max_len, a.limit, a.country, a.no_test, view=a.view)


def _run(args, tok, out):
    """Body of run(): zone pairs, texts, two cross-fitted models, scores."""
    tr, te = zone_pairs(args.work, args.model, args.zone, args.country)
    if args.no_test:
        te = te.head(0)
    if args.limit:
        tr = tr.sample(min(args.limit, tr.height), seed=1)
        te = te.sample(min(args.limit, te.height), seed=1)
    log(f"zone {args.zone} < p < {1 - args.zone}: train {tr.height:,} pairs ({tr['label'].sum():,} true), "
        f"test {te.height:,} pairs")
    tx_tr = texts(args.data, "train", pl.concat([tr["s1_id"], tr["cand_id"]]).unique())
    tx_te = texts(args.data, "test", pl.concat([te["s1_id"], te["cand_id"]]).unique())
    a_tr, b_tr = [tx_tr[i] for i in tr["s1_id"]], [tx_tr[i] for i in tr["cand_id"]]
    a_te, b_te = [tx_te[i] for i in te["s1_id"]], [tx_te[i] for i in te["cand_id"]]
    del tx_tr, tx_te
    fold, y = tr["fold"].to_numpy(), tr["label"].to_numpy()
    log(f"texts ready; example: {a_tr[0]!r} vs {b_tr[0]!r} (label {y[0]})")

    oof = np.full(tr.height, np.nan, dtype=np.float32)
    test = np.zeros(te.height, dtype=np.float32)
    for k in (0, 1):
        pf = os.path.join(out, f"fold{k}_pred.npz")
        if os.path.exists(pf):
            z = np.load(pf)
            oof[fold == 1 - k], test = z["oof"], test + z["test"] / 2
            log(f"fold {k}: loaded {pf}")
            continue
        tri = np.flatnonzero(fold == k)
        log(f"fold {k}: training on {len(tri):,} pairs ({int(y[tri].sum()):,} true), {args.epochs} epoch(s)")
        m = train(tok, [a_tr[j] for j in tri], [b_tr[j] for j in tri], y[tri], args, seed=17 + k)
        torch.save(m.state_dict(), os.path.join(out, f"fold{k}.pt"))
        ev = np.flatnonzero(fold == 1 - k)
        log(f"fold {k}: scoring {len(ev):,} out-of-fold pairs")
        po = predict(m, tok, [a_tr[j] for j in ev], [b_tr[j] for j in ev], 256, args.max_len)
        log(f"fold {k}: scoring {te.height:,} test pairs")
        pt = predict(m, tok, a_te, b_te, 256, args.max_len) if te.height else np.zeros(0, dtype=np.float32)
        np.savez(pf, oof=po, test=pt)
        oof[ev], test = po, test + pt / 2
        p = 1 / (1 + np.exp(-po))
        yy = y[ev]
        ll = -np.mean(yy * np.log(np.clip(p, 1e-7, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-7, 1)))
        p2 = np.clip(tr["prob"].to_numpy()[ev], 1e-7, 1 - 1e-7)
        ll2 = -np.mean(yy * np.log(p2) + (1 - yy) * np.log(1 - p2))
        log(f"fold {k}: out-of-fold log-loss cross-encoder {ll:.4f} vs stage 2 {ll2:.4f}; "
            f"accuracy {np.mean((p > 0.5) == yy):.4f} vs {np.mean((p2 > 0.5) == yy):.4f}")
        del m
        torch.cuda.empty_cache()
    tr.select("s1_id", "cand_id").with_columns(pl.Series("ce_logit", oof)).write_parquet(os.path.join(out, "train_scores.parquet"))
    te.select("s1_id", "cand_id").with_columns(pl.Series("ce_logit", test)).write_parquet(os.path.join(out, "test_scores.parquet"))
    log("cross-encoder finished")


if __name__ == "__main__":
    main()
