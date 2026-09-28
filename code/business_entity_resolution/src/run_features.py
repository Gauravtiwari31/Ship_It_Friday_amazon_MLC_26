"""Stage 3 driver: compute pair features for every candidate pair of a split.

Writes <work>/<split>_feats/part_XXX.parquet (float32 features, plus `label` for train).
"""
import glob
import os
import sys
import time

import polars as pl

from features import context_features, pair_features


def fold_of(ids):
    """Deterministic 2-fold assignment of S1 ids (used for cross-fitting)."""
    return ids.str.extract(r"(\d+)$").cast(pl.Int64) % 2


def run(work_dir, split, chunk=2_000_000, log=print):
    """Compute the pair features of a split (stage 3) and write them as row-aligned parts in
    <work_dir>/<split>_feats/, sorted by S1 id, with the cross-fitting fold and (train) the label."""
    out_dir = os.path.join(work_dir, f"{split}_feats")
    done = os.path.join(out_dir, "_DONE")
    parts = glob.glob(os.path.join(out_dir, "*.parquet"))
    if parts and os.path.exists(done):
        return out_dir
    for p in parts:  # an interrupted run: start the split again
        os.remove(p)
    os.makedirs(out_dir, exist_ok=True)
    t = time.time()
    cands = pl.read_parquet(os.path.join(work_dir, f"{split}_cands.parquet"))
    cands = context_features(cands).with_columns(fold_of(pl.col("s1_id")).alias("fold"))
    if split == "train":
        gt = pl.read_parquet(os.path.join(work_dir, "train_gt_pairs.parquet")).with_columns(pl.lit(1, pl.Int8).alias("label"))
        cands = cands.join(gt, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("label").fill_null(0))
    # sort so that each part holds whole S1 groups (keeps per-S1 logic simple)
    cands = cands.sort("s1_id")
    norm = pl.read_parquet(os.path.join(work_dir, f"{split}_norm.parquet"))
    for i, s in enumerate(range(0, cands.height, chunk)):
        part = pair_features(cands.slice(s, chunk), norm)
        part.write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        log(f"[{time.time() - t:6.0f}s] {split} features {min(s + chunk, cands.height):,}/{cands.height:,}")
    open(done, "w").close()
    return out_dir


if __name__ == "__main__":
    work = sys.argv[1]
    for split in sys.argv[2:]:
        run(work, split, log=lambda m: print(m, flush=True))
