"""Stage 2b: address-component equivalences for countries without training labels (v6).

The hand-written state tables cover US and Indian states, which are moved out of the
address tokens into a separate state field. For a country absent from the training labels
(France) nothing is detected, so region / department names stay in the address tokens. In
the test data a French S1 address ends with its region ("..., Bordeaux, Nouvelle-Aquitaine")
while a matching S2/S3 record often carries the department ("..., BORDEAUX, Gironde") or no
region at all, so true matches look like partial address mismatches (v4 predicted matches:
address token-set similarity q25 = 90 in France vs 98-100 in the US / India).

The equivalences are learned from the data itself, without labels or a model:
  * anchor pairs: candidate pairs of the country with the same core name and the same first
    house number (almost surely the same business);
  * comma components present on only one side of an anchor are aligned; a component is
    linked to its most frequent partner when that partner accounts for >= LINK_SHARE of its
    unmatched occurrences (e.g. "gironde" -> "nouvelle aquitaine", "st nazaire" -> "saint nazaire");
  * linked components form classes named after their most frequent member;
  * a class whose canonical member co-occurs with >= HUB distinct frequent components (a region
    co-occurs with many cities) is an administrative area and becomes the record's state, as
    for US / Indian states; other classes are spelling variants mapped to the canonical form.
"""
import json
import os
from collections import Counter

import polars as pl

from normalize import addr_components

MIN_LINK = 200       # anchor pairs needed to link two components
LINK_SHARE = 0.9     # the partner must explain this share of a component's unmatched occurrences
HUB = 4              # distinct frequent co-occurring components that make a class an admin area
FREQ = 0.002         # "frequent" component: in >= 0.2% of the country's records


def _comps(raw, tmap):
    """Set of digit-free normalised components of an address."""
    return {c for c in addr_components(raw or "", tmap) if not any(ch.isdigit() for ch in c)}


def learn_component_map(raw, norm, cands, countries, tmap, log=print):
    """Learn {component: state code} and {component: canonical spelling} for `countries`.

    raw: entity_id, business_address, country; norm: entity_id, name_core, addr_nums;
    cands: s1_id, cand_id candidate pairs."""
    state, alias, report = {}, {}, []
    for country in countries:
        ids = raw.filter(pl.col("country") == country)
        addr = dict(zip(ids["entity_id"].to_list(), ids["business_address"].to_list()))
        key = norm.filter(pl.col("entity_id").is_in(ids["entity_id"].implode())).select(
            "entity_id", "name_core", pl.col("addr_nums").str.split(" ").list.first().alias("n1"))
        a = key.rename({"entity_id": "s1_id", "name_core": "na", "n1": "ka"})
        b = key.rename({"entity_id": "cand_id", "name_core": "nb", "n1": "kb"})
        anchors = (cands.select("s1_id", "cand_id").join(a, on="s1_id").join(b, on="cand_id")
                   .filter((pl.col("na") == pl.col("nb")) & (pl.col("na") != "")
                           & (pl.col("ka") == pl.col("kb")) & (pl.col("ka") != "")))
        pair, single = Counter(), Counter()
        for s1, c in anchors.select("s1_id", "cand_id").iter_rows():
            A, B = _comps(addr.get(s1), tmap), _comps(addr.get(c), tmap)
            if not A or not B:
                continue
            da, db = A - B, B - A
            for x in da | db:
                single[x] += 1
            for x in da:
                for y in db:
                    pair[(x, y)] += 1
                    pair[(y, x)] += 1
        best = {}
        for (x, y), k in pair.items():
            if k > best.get(x, (None, 0))[1]:
                best[x] = (y, k)
        parent = {}

        def find(x):
            """Union-find root of x."""
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for x, (y, k) in best.items():
            if k >= MIN_LINK and k >= LINK_SHARE * single[x]:
                parent[find(x)] = find(y)
        classes = {}
        for x in {x for x in parent} | {parent[x] for x in parent}:
            classes.setdefault(find(x), set()).add(x)
        if not classes:
            continue
        # component frequencies and co-occurrence among frequent components of the country
        freq, co = Counter(), {}
        comps_per_rec = [_comps(v, tmap) for v in addr.values()]
        for cs in comps_per_rec:
            freq.update(cs)
        frequent = {c for c, n in freq.items() if n >= FREQ * len(comps_per_rec)}
        for cs in comps_per_rec:
            f = cs & frequent
            for x in f:
                co.setdefault(x, set()).update(f - {x})
        for members in classes.values():
            canon = max(members, key=lambda m: freq[m])
            hub = len(co.get(canon, set()) - members)
            kind = "state" if hub >= HUB else "alias"
            for m in members:
                if kind == "state":
                    state[m] = canon.replace(" ", "_")
                elif m != canon:
                    alias[m] = canon
            report.append(f"{country}: {kind:5s} {canon!r} <- {sorted(members - {canon})} (hub {hub})")
        log(f"  {country}: {anchors.height:,} anchor pairs, {len(classes)} component classes")
    for r in report:
        log("    " + r)
    return {"addr_state_learned": state, "addr_alias_learned": alias}


def refine_unseen_addresses(work, tmap, log=print):
    """Learn the component maps for test countries absent from the training labels and
    re-normalise the address fields of their records in <work>/test_norm.parquet.

    The unrefined table is kept as test_norm_base.parquet; test feature caches built from it
    are renamed (*_base) so that stage 3 rebuilds them. Idempotent: the learned maps are saved
    in <work>/model/addr_component_map.json and the step is skipped when it exists."""
    from normalize import addr_features

    out_map = os.path.join(work, "model", "addr_component_map.json")
    if os.path.exists(out_map):
        with open(out_map, encoding="utf-8") as f:
            return json.load(f)
    norm_path, base_path = os.path.join(work, "test_norm.parquet"), os.path.join(work, "test_norm_base.parquet")
    if not os.path.exists(base_path):
        os.replace(norm_path, base_path)
    base = pl.read_parquet(base_path)
    seen = set(pl.read_parquet(os.path.join(work, "train_norm.parquet"), columns=["src", "country"])
               .filter(pl.col("src") == 1)["country"].unique().to_list())
    unseen = sorted(set(base.filter(pl.col("src") == 1)["country"].unique().to_list()) - seen)
    raw = pl.read_parquet(os.path.join(work, "test_raw.parquet"), columns=["entity_id", "business_address", "country"])
    cands = pl.read_parquet(os.path.join(work, "test_cands.parquet"), columns=["s1_id", "cand_id"])
    maps = learn_component_map(raw, base, cands, unseen, tmap, log=log)

    tm = {**tmap, **maps}
    sub = raw.filter(pl.col("country").is_in(unseen))
    feats = [addr_features(a or "", tm) for a in sub["business_address"].to_list()]
    cols = ["addr_tokens", "addr_words", "addr_nums", "addr_state", "addr_empty"]
    new = pl.DataFrame({"entity_id": sub["entity_id"], **{c: [f[c] for f in feats] for c in cols}})
    refined = base.join(new, on="entity_id", how="left", suffix="_new", maintain_order="left").with_columns(
        [pl.coalesce(pl.col(c + "_new"), pl.col(c)).alias(c) for c in cols]).select(base.columns)
    changed = refined.filter(pl.col("addr_tokens") != base["addr_tokens"]).height
    refined.write_parquet(norm_path)
    log(f"  re-normalised {sub.height:,} records of {unseen}; address tokens changed for {changed:,}")
    for d in ("test_feats", "test_extra", "test_numrel"):
        p = os.path.join(work, d)
        if os.path.isdir(p) and not os.path.isdir(p + "_base"):
            os.replace(p, p + "_base")
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=1)
    return maps
