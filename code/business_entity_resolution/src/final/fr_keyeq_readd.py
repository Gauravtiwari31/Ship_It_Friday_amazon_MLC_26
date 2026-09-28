"""France decision: flat 0.95 (one owner per record) + a train-validated high-precision re-add. A pair rejected at 0.95 but
kept by the rank rule 0.60 / 0.75 is re-added only when BOTH exact keys agree: f_name_key_eq (order-free core-name tokens
equal) and f_addr_key_eq (order-free address tokens equal, address non-empty). No other gate; nothing is fitted on test.
Leave-one-country-out at corrected 2x (source-only training): unseen India 495 re-added / 493 true (F0.5 0.96746 → 0.96792),
unseen US 277 / 276 (0.98283 → 0.98293).
--loo: replicate those LOO numbers from work/fr_claude/top2/loo_fin_*_2x_base.parquet.
usage: fr_keyeq_readd.py [--fin work/fr_claude/fr_test_final_bi_flat95.parquet] [--tag bi_flat95_keyeq] [--loo]"""
import argparse, hashlib, os, sys
import numpy as np, polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
from decide import assign_owner, keep_by_rank
from train import part_files
ap = argparse.ArgumentParser(); ap.add_argument("--fin", default="work/fr_claude/fr_test_final_bi_flat95.parquet")
ap.add_argument("--tag", default="bi_flat95_keyeq"); ap.add_argument("--loo", action="store_true"); a = ap.parse_args()
K = ["s1_id", "cand_id"]; E = ["f_name_key_eq", "f_addr_key_eq"]; ORIG = pl.col("cand_id").str.replace(r"#d\d+$", "")
def keys(work, split, pairs):  # exact-key flags of the given pairs (extra parts are row-aligned with the feature parts)
    out = []
    for f, e in zip(part_files(f"{work}/{split}_feats"), part_files(f"{work}/{split}_extra")):
        out.append(pl.concat([pl.read_parquet(f, columns=K), pl.read_parquet(e, columns=E)], how="horizontal").join(pairs, on=K, how="semi"))
    return pl.concat(out)
def decide(fin, work, split):
    owned = assign_owner(fin); base = keep_by_rank(owned, 0.95, 0.95).select(K)
    band = keep_by_rank(owned, 0.60, 0.75).select(K).join(base, on=K, how="anti")
    add = keys(work, split, band).filter((pl.col("f_name_key_eq") == 1) & (pl.col("f_addr_key_eq") == 1)).select(K)
    return base, band, add
if a.loo:
    from evaluate import macro_f05
    W = "work_lg"; gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet").select(K)
    s1c = pl.read_parquet(f"{W}/train_norm.parquet", columns=["entity_id", "src", "country"]).filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1_id"), "country")
    rng = np.random.default_rng(5)
    SAMPLE = {c: set(g["s1_id"].to_numpy()[rng.choice(g.height, min(250000, g.height), replace=False)]) for (c,), g in s1c.sort("s1_id").group_by("country", maintain_order=True)}
    for d, C in (("UStoIndia", "India"), ("IndiatoUS", "US")):
        fin = pl.read_parquet(f"work/fr_claude/top2/loo_fin_{d}_2x_base.parquet"); base, band, add = decide(fin, "work_lg_2x", "train")
        s1 = list(SAMPLE[C]); gs = gt.join(pl.DataFrame({"s1_id": s1}), on="s1_id", how="semi")
        y = add.with_columns(ORIG.alias("_o")).join(gs.rename({"cand_id": "_o"}).with_columns(pl.lit(1).alias("y")), on=["s1_id", "_o"], how="left")["y"].fill_null(0)
        f0, f1 = macro_f05(base, gs, s1)["macro_f05"], macro_f05(pl.concat([base, add]), gs, s1)["macro_f05"]
        print(f"unseen {C} 2x: band {band.height:,}; re-added {add.height:,}, true {int(y.sum()):,} (precision {y.mean():.5f}); F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)
    sys.exit(0)
fin = pl.read_parquet(a.fin); base, band, add = decide(fin, "work_lg", "test")
m = pl.concat([base, add])
tn = pl.read_parquet("work_lg/test_norm.parquet", columns=["entity_id", "src", "country"])
FRS = set(tn.filter((pl.col("src") == 1) & (pl.col("country") == "France"))["entity_id"].to_list())
assert set(fin["s1_id"].unique().to_list()) <= FRS
order = [s for s in pl.read_csv("dataset/test/test_source1.tsv", separator="\t", columns=["entity_id"], quote_char=None)["entity_id"].to_list() if s in FRS]
def mp(d): return dict(d.sort(K).group_by("s1_id", maintain_order=True).agg("cand_id").iter_rows())
os.makedirs("submission/fr_claude", exist_ok=True)
for fn, col, d in ((f"candidate_{a.tag}_fr.tsv", "candidate_entity_ids", mp(fin.select(K))), (f"matching_{a.tag}_fr.tsv", "matched_entity_ids", mp(m))):
    with open(f"submission/fr_claude/{fn}", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in order: fh.write(f"{s}\t{','.join(d.get(s, []))}\n")
    print(fn, "sha256", hashlib.sha256(open(f"submission/fr_claude/{fn}", "rb").read()).hexdigest())
print(f"France S1 {len(order):,}; candidates {fin.height:,}; flat-0.95 matches {base.height:,}; band {band.height:,}; "
      f"re-added (name key = address key = equal) {add.height:,} in {add['s1_id'].n_unique():,} S1; total matches {m.height:,}")
