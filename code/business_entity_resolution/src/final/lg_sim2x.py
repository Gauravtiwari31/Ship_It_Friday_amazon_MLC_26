"""leader_gap: corrected-2x density pool for work_lg (retrieval-augmented candidates), via src/final/sim_density.py
with the bienc rescue on. Distractor clones inherit their original's retrieval candidate (identical text -> identical
top-1). Phase 1 (norm,block,feat,extra,numrel) waits for work_lg normalisation + retrieval output + work_lg train blocking;
phase 2 (s1,ctx: frozen work_lg stage-1 models, de-twinned contexts) waits for work_lg/model_v9/stage1_fold1.txt.
usage: lg_sim2x.py <phase: 1|2>   (all code under the __main__ guard: Windows spawn re-imports this module)"""
import os, sys, time


def wait(paths):
    while not all(os.path.exists(p) for p in paths):
        time.sleep(20)


def main():
    sys.path.insert(0, "code/business_entity_resolution/src")
    import config
    config.RESCUE = {**config.RESCUE, "bienc": {"path": "bienc_cands_{split}.parquet", "k": 1}}
    import polars as pl
    W, S = "work_lg", "work_lg_2x"; os.makedirs(S, exist_ok=True)
    phase = sys.argv[1]
    if phase == "1":
        wait([f"{W}/train_norm.parquet", f"{W}/train_gt_pairs.parquet", "work/audit/leader_gap/retrieve_done.txt", f"{W}/train_cands.parquet"])
        b = pl.read_parquet("work/audit/leader_gap/bienc_cands_train.parquet")
        gt = pl.read_parquet(f"{W}/train_gt_pairs.parquet", columns=["cand_id"]).unique()
        dis = b.join(gt, on="cand_id", how="anti")
        pl.concat([b, dis.with_columns((pl.col("cand_id") + "#d1").alias("cand_id"))]).write_parquet(f"{S}/bienc_cands_train.parquet")
        print(f"bienc for the 2x pool: {b.height:,} + {dis.height:,} clone rows", flush=True)
        steps = "norm,block,feat,extra,numrel"
    else:
        wait([f"{W}/model_v9/stage1_fold1.txt"])
        steps = "s1,ctx"
    sys.argv = ["sim_density.py", W, S, "dataset", "--factor", "2", "--detwin", "--steps", steps]
    exec(open("code/business_entity_resolution/src/final/sim_density.py", encoding="utf-8").read(), {"__name__": "sim_density_exec"})
    print("PHASE DONE", phase, flush=True)


if __name__ == "__main__":
    main()
