"""Final step: merge the India+US rows (iu_export.py) and the France rows (fr_meta_prod.py --tag bi_flat95) into
output/matching_results.tsv and output/candidate_pairs.tsv (one row per test Source-1 entity, in test_source1.tsv order),
then run the official validator when it is available.
usage: make_final.py [--out output] [--fr-tag bi_flat95]"""
import argparse, hashlib, os, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="output"); ap.add_argument("--fr-tag", default="bi_flat95"); a = ap.parse_args()
def rows(p):
    with open(p, encoding="utf-8") as f:
        f.readline(); return dict(l.rstrip("\n").split("\t") for l in f)
iu_m, iu_c = rows("submission/iu_claude/matching_results_iu.tsv"), rows("submission/iu_claude/candidate_pairs_iu.tsv")
fr_m, fr_c = rows(f"submission/fr_claude/matching_{a.fr_tag}_fr.tsv"), rows(f"submission/fr_claude/candidate_{a.fr_tag}_fr.tsv")
assert not set(iu_m) & set(fr_m), "India+US and France rows overlap"
order = [l.split("\t", 1)[0] for l in open("dataset/test/test_source1.tsv", encoding="utf-8").read().split("\n")[1:] if l]
M, C = {**iu_m, **fr_m}, {**iu_c, **fr_c}
assert len(M) == len(C) == len(order) and set(M) == set(order), (len(M), len(C), len(order))
for s in order:  # every match must be a candidate
    if M[s]:
        assert set(M[s].split(",")) <= set(C[s].split(",")), s
os.makedirs(a.out, exist_ok=True)
for fn, h, d in (("matching_results.tsv", "source1_entity_id\tmatched_entity_ids\n", M), ("candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\n", C)):
    with open(f"{a.out}/{fn}", "w", encoding="utf-8", newline="\n") as f:
        f.write(h + "\n".join(f"{s}\t{d[s]}" for s in order) + "\n")
    print(fn, "sha256", hashlib.sha256(open(f"{a.out}/{fn}", "rb").read()).hexdigest())
n = lambda d: sum(len(v.split(",")) for v in d.values() if v)
print(f"S1 {len(order):,}; candidates {n(C):,} ({n(C) / len(order):.3f}/S1); matches {n(M):,}")
if os.path.exists("utils/validate_submission.py"):
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", f"{a.out}/matching_results.tsv", "--candidate",
                        f"{a.out}/candidate_pairs.tsv", "--test-dir", "dataset/test", "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-300:])
