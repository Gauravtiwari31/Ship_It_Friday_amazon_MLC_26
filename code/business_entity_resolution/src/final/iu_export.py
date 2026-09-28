"""iu_claude: export India+US-only test artifacts from a test final-probability table (IU S1 only).
Writes <outdir>/matching_results_iu.tsv, <outdir>/candidate_pairs_iu.tsv (one row per India/US test S1; candidates =
exactly the pairs the matcher scored) and <scores> (s1_id, cand_id, prob). Checks: IU S1 only, every S1 once, matches ⊆
candidates, ids exist in the test S2/S3 files, no duplicates. Refuses to overwrite.
usage: iu_export.py <test_final parquet> <top> <rest> <outdir> <scores parquet>"""
import sys, os, hashlib
import polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
from decide import assign_owner, keep_by_rank
K = ["s1_id", "cand_id"]
src, top, rest, outdir, scores = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
for f in ("matching_results_iu.tsv", "candidate_pairs_iu.tsv"):
    assert not os.path.exists(os.path.join(outdir, f)), f"refusing to overwrite {f}"
assert not os.path.exists(scores), "refusing to overwrite scores"
os.makedirs(outdir, exist_ok=True)
tn = pl.read_parquet("work_v17r/test_norm.parquet", columns=["entity_id", "src", "country"])
iu = tn.filter((pl.col("src") == 1) & pl.col("country").is_in(["India", "US"]))["entity_id"].to_list()
fin = pl.read_parquet(src, columns=[*K, "prob"])
assert fin["s1_id"].is_in(pl.Series(iu).implode()).all(), "non-IU S1 in input"
fin.write_parquet(scores)
m = keep_by_rank(assign_owner(fin), top, rest).select(K)
valid = set(tn.filter(pl.col("src") != 1)["entity_id"].to_list())
assert set(fin["cand_id"].unique().to_list()) <= valid, "unknown candidate ids"
def mp(df): return dict(df.sort(K).group_by("s1_id", maintain_order=True).agg("cand_id").iter_rows())
C, M = mp(fin.select(K).unique(K)), mp(m)
IUS = set(iu)
order = [s for s in pl.read_csv("dataset/test/test_source1.tsv", separator="\t", columns=["entity_id"], quote_char=None)["entity_id"].to_list() if s in IUS]
for fn, col, d in (("candidate_pairs_iu.tsv", "candidate_entity_ids", C), ("matching_results_iu.tsv", "matched_entity_ids", M)):
    with open(os.path.join(outdir, fn), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in order: f.write(f"{s}\t{','.join(d.get(s, []))}\n")
print(f"IU S1 {len(order):,}; candidates {fin.height:,} ({fin.height/len(order):.3f}/S1); matches {m.height:,}; rule {top}/{rest}")
for fn in ("matching_results_iu.tsv", "candidate_pairs_iu.tsv"):
    print(fn, hashlib.sha256(open(os.path.join(outdir, fn), "rb").read()).hexdigest())
