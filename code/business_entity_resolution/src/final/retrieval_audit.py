"""Reconstruct CE3+DBA exactly; count current errors and explicitly optimistic ceilings.

Only labelled training data and frozen out-of-fold scores. No test labels/data.
All artifacts are new; prior champions and artifacts remain unchanged.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.cpu_count = lambda: 6
import gc
import json
import sys
import time
from pathlib import Path
import polars as pl
sys.path.insert(0, "code/business_entity_resolution/src")
from decide import assign_owner, keep_by_rank
from train import load_model, model_path, score_parts

OUT = Path("work/audit/ceiling6h/retrieval")
OUT.mkdir(parents=True, exist_ok=True)
K = ["s1_id", "cand_id"]
T0 = time.time()
def log(s): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:.0f}s] {s}", flush=True)

gt = pl.read_parquet("work_v17r/train_gt_pairs.parquet").select(K)
norm = pl.read_parquet("work_v17r/train_norm.parquet", columns=["entity_id", "src", "country", "name_core", "addr_empty"])
nk = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ")
s1 = norm.filter(pl.col("src") == 1).with_columns(nk.alias("nk"))
s1 = s1.join(s1.group_by("country", "nk").len("name_owners"), on=["country", "nk"]).select(pl.col("entity_id").alias("s1_id"), "country", "nk", "name_owners")
attr = norm.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("cand_id"), pl.col("src").alias("cand_src"), "addr_empty")
del norm
nt = gt.group_by("s1_id").len("nt")

def metric(pred):
    tp = pred.join(gt, on=K, how="semi").group_by("s1_id").len("tp")
    d = s1.select("s1_id", "country").join(nt, on="s1_id", how="left").join(tp, on="s1_id", how="left").join(pred.group_by("s1_id").len("np"), on="s1_id", how="left").fill_null(0)
    d = d.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.).otherwise(1.25*pl.col("tp")/(.25*pl.col("nt")+pl.col("np"))).alias("f05"))
    ntrue, npred, ntp = int(d['nt'].sum()), int(d['np'].sum()), int(d['tp'].sum())
    return dict(F05=d['f05'].mean(), precision=ntp/max(npred,1), recall=ntp/max(ntrue,1), FP=npred-ntp,FN=ntrue-ntp,TP=ntp, **{c:d.filter(pl.col('country')==c)['f05'].mean() for c in ['India','US']})

models = [load_model(model_path("work/audit/099/s3ce3/model", "stage3_ce3", k)) for k in (0,1)]
all_summary=[]
for density, work, zone, ce3 in [
    ("1x", "work_v17r", "work_v17r/train_zone_ctx_ce2norm_model_v9", "work/audit/099/s3ce3/train_zone_ce3"),
    ("2x", "work_v17r_2x", "work_v17r_2x/train_zone_ctx_ce2norm", "work/audit/099/s3ce3/train_zone_ce3_2x")]:
    path=OUT/f"champion_oof_{density}.parquet"
    if path.exists():
        scored=pl.read_parquet(path)
    else:
        p3=score_parts([zone,ce3],models,oof=True,log=log).select(*K,pl.col('prob').alias('_p3'))
        scored=pl.read_parquet(f"{work}/train_oof_model_v9.parquet",columns=[*K,'prob']).join(p3,on=K,how='left',maintain_order='left').with_columns(pl.coalesce('_p3','prob').alias('prob')).drop('_p3')
        del p3
        scored.write_parquet(path)
        log(f"wrote exact {density} champion OOF: {scored.height:,}")
    owned=assign_owner(scored)
    # Runner-up includes all owners; margin uses final score and one-owner winner.
    second=scored.group_by('cand_id').agg(pl.col('prob').top_k(2).sort(descending=True).get(1,null_on_oob=True).fill_null(0.).alias('p_second'))
    owned=owned.join(second,on='cand_id').with_columns((pl.col('prob')-pl.col('p_second')).alias('owner_margin'))
    del second
    pred=keep_by_rank(owned,.60,.75).select(K)
    pred.write_parquet(OUT/f'champion_accepted_{density}.parquet')
    anchors=owned.filter((pl.col('prob')>=.995)&(pl.col('owner_margin')>=.3))
    anchors.write_parquet(OUT/f'champion_confident_{density}.parquet')
    owned.write_parquet(OUT/f'champion_owned_{density}.parquet')
    base=metric(pred)
    log(f'{density} baseline {base}, confident anchors {anchors.height:,}')
    fp=pred.join(gt,on=K,how='anti').join(nt,on='s1_id',how='left').join(gt.select('cand_id',pl.col('s1_id').alias('true_owner')),on='cand_id',how='left').with_columns(pl.when(pl.col('nt').is_null()).then(pl.lit('singleton_false_positive')).when(pl.col('true_owner').is_not_null()).then(pl.lit('wrong_winner')).otherwise(pl.lit('accepted_false_positive')).alias('class'))
    fn=gt.join(pred,on=K,how='anti').join(s1,on='s1_id').join(attr,on='cand_id').join(scored.select(*K,'prob',pl.lit(True).alias('in_candidate')),on=K,how='left').join(owned.select('cand_id',pl.col('s1_id').alias('predicted_owner'),pl.col('prob').alias('best_prob')),on='cand_id',how='left').with_columns(pl.col('in_candidate').fill_null(False),(pl.col('addr_empty')&(pl.col('name_owners')>=2)).alias('old_ambiguous'))
    fn=fn.with_columns(pl.when(~pl.col('in_candidate')).then(pl.lit('blocking_false_negative')).when(pl.col('predicted_owner')!=pl.col('s1_id')).then(pl.lit('multi_owner_ranking_conflict')).otherwise(pl.lit('in_candidate_false_negative')).alias('class'))
    support=anchors.join(attr,on='cand_id').group_by('s1_id').agg(pl.len().alias('confident_siblings'),pl.col('cand_src').n_unique().alias('support_sources'),(pl.col('cand_src')==2).sum().alias('support_s2'),(pl.col('cand_src')==3).sum().alias('support_s3'))
    fn=fn.join(support,on='s1_id',how='left').with_columns(pl.col(['confident_siblings','support_sources','support_s2','support_s3']).fill_null(0))
    fp.write_parquet(OUT/f'fp_{density}.parquet');fn.write_parquet(OUT/f'fn_{density}.parquet')
    rows=[]
    for row in fn.group_by('class','old_ambiguous').agg(pl.len().alias('count'),(pl.col('confident_siblings')>0).sum().alias('with_sibling'),(pl.col('support_sources')>=2).sum().alias('with_two_sources')).iter_rows(named=True):
        d=fn.filter((pl.col('class')==row['class'])&(pl.col('old_ambiguous')==row['old_ambiguous'])).select(K)
        oracle=metric(pl.concat([pred,d]).unique(K))
        rows.append(dict(**row,oracle_F05=oracle['F05'],oracle_gain=oracle['F05']-base['F05']))
    for row in fp.group_by('class').len('count').iter_rows(named=True):
        oracle=metric(pred.join(fp.filter(pl.col('class')==row['class']).select(K),on=K,how='anti'))
        rows.append(dict(**row,old_ambiguous=False,with_sibling=None,with_two_sources=None,oracle_F05=oracle['F05'],oracle_gain=oracle['F05']-base['F05']))
    pl.DataFrame(rows).write_csv(OUT/f'error_buckets_{density}.csv')
    # These are label oracles, NOT validated achievable scores.
    valid=pred.join(gt,on=K,how='semi')
    choices=[('1_perfect_existing_candidate_decisions',gt.join(scored.select(K),on=K,how='semi')),
             ('2_fix_nonambiguous_in_candidate_and_all_FP',pl.concat([valid,fn.filter(pl.col('in_candidate')&~pl.col('old_ambiguous')).select(K)]).unique(K)),
             ('3_fix_all_old_nonambiguous_and_all_FP',pl.concat([valid,fn.filter(~pl.col('old_ambiguous')).select(K)]).unique(K)),
             ('4_optimistic_ambiguous_with_any_confident_sibling',pl.concat([valid,fn.filter(~pl.col('old_ambiguous')|(pl.col('confident_siblings')>0)).select(K)]).unique(K))]
    ceilings={name:metric(p)['F05'] for name,p in choices}
    result=dict(density=density,baseline=base,candidates=scored.height,anchors=anchors.height,ceilings=ceilings,buckets=rows)
    json.dump(result,open(OUT/f'audit_{density}.json','w'),indent=2)
    log(f'{density} ceilings (4 is opportunity bound only): {ceilings}')
    all_summary.append(result)
    del scored,owned,pred,anchors,fp,fn,support,valid,choices
    gc.collect()
json.dump(all_summary,open(OUT/'audit_summary.json','w'),indent=2)
log('DONE')
