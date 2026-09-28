"""Stage 2 driver: candidate generation for a split (writes <work>/<split>_cands.parquet)."""
import os
import sys
import time

import polars as pl

from blocking import generate_candidates
from config import BLOCKING, RESCUE


def run(work_dir, split, log=print):
    """Generate the candidate pairs of a split (stage 2), cached as <work_dir>/<split>_cands.parquet."""
    out = os.path.join(work_dir, f"{split}_cands.parquet")
    if os.path.exists(out):
        return pl.read_parquet(out)
    t = time.time()
    norm = pl.read_parquet(os.path.join(work_dir, f"{split}_norm.parquet"))
    rescue = RESCUE
    if rescue and "bienc" in rescue:  # learned-retrieval candidates are precomputed per split (retrieval.py)
        rescue = {**rescue, "bienc": {**rescue["bienc"], "path": os.path.join(work_dir, rescue["bienc"]["path"].format(split=split))}}
    cands = generate_candidates(norm, BLOCKING, log=lambda m: log(f"[{time.time() - t:6.0f}s] {m}"), rescue=rescue)
    cands.write_parquet(out)
    return cands


def blocking_recall(cands, gt_pairs, s1_ids=None):
    """Share of the true (S1, candidate) pairs that are in the candidate set (optionally for some S1 only)."""
    g = gt_pairs if s1_ids is None else gt_pairs.filter(pl.col("s1_id").is_in(s1_ids))
    hit = g.join(cands.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi").height
    return hit / max(1, g.height)


if __name__ == "__main__":
    work = sys.argv[1]
    for split in sys.argv[2:]:
        c = run(work, split)
        n1 = c["s1_id"].n_unique()
        print(f"{split}: {c.height:,} pairs, {n1:,} S1 with candidates, {c.height / max(1, n1):.1f} per S1", flush=True)
        if split == "train":
            gt = pl.read_parquet(os.path.join(work, "train_gt_pairs.parquet"))
            print(f"train blocking pair recall: {blocking_recall(c, gt):.4f}", flush=True)
