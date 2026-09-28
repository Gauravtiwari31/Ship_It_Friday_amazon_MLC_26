"""leader_gap: production pipeline (src/run_pipeline.py) in a fresh work folder with the learned-retrieval rescue.
In-memory config overrides only (src/config.py untouched):
  RESCUE += bienc (record-centric top-1 retrieval candidates, precomputed by src/final/retrieve_cands.py),
  UNSEEN_SELFTRAIN = None (no test self-training: user ruling), USE_CE = False (stage 3 is rebuilt by the cascade scripts).
Blocking waits until <work>/bienc_cands_{split}.parquet exists (copied from work/audit/leader_gap/).
All code runs under the __main__ guard: Windows multiprocessing (spawn) re-imports this module in every worker."""
import os, sys, time, shutil


def main():
    sys.path.insert(0, "code/business_entity_resolution/src")
    work = sys.argv[1] if len(sys.argv) > 1 else "work_lg"
    os.makedirs(work, exist_ok=True)
    import config
    config.RESCUE = {**config.RESCUE, "bienc": {"path": "bienc_cands_{split}.parquet", "k": 1}}
    config.UNSEEN_SELFTRAIN = None
    config.USE_CE = False
    import run_blocking
    orig_run = run_blocking.run

    def run(work_dir, split, log=print):
        dst = os.path.join(work_dir, f"bienc_cands_{split}.parquet"); src = f"work/audit/leader_gap/bienc_cands_{split}.parquet"
        while not os.path.exists(dst):
            if os.path.exists(src) and os.path.exists("work/audit/leader_gap/retrieve_done.txt"):
                shutil.copyfile(src, dst); break
            time.sleep(20)
        return orig_run(work_dir, split, log=log)
    run_blocking.run = run
    import run_pipeline
    sys.argv = ["run_pipeline.py", "--data", "dataset", "--work", work, "--out", os.path.join(work, "out_stage2")]
    run_pipeline.main()


if __name__ == "__main__":
    main()
