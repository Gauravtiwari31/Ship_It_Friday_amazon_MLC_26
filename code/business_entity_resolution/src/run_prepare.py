"""Run stage 1 on its own: python src/run_prepare.py <data dir> <work dir> <model dir> train test

Learns the transliteration map and writes <work dir>/<split>_norm.parquet for each split given."""
import sys, time
from prepare import build_translit, normalize_split
if __name__ == "__main__":
    data, work, model = sys.argv[1], sys.argv[2], sys.argv[3]
    t = time.time()
    tmap = build_translit(data, work, model)
    print("translit", len(tmap["tok"]), len(tmap["comp"]), round(time.time() - t, 1), flush=True)
    for split in sys.argv[4:]:
        df = normalize_split(data, split, work, tmap)
        print(split, df.shape, round(time.time() - t, 1), flush=True)
