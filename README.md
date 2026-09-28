# Ship It Friday — ML Challenge 2026: Business Entity Resolution

**Team:** Ship It Friday — Gaurav Tiwari (Lead), Harshi Singh, Tushar Singh Akhawat, Prajjwal Singh

A record-centric blocking → learned-retrieval → cascaded-matching pipeline that links Source-2/3 business
records to their Source-1 entity. Multi-view TF-IDF blocking is paired with a supervised multilingual
bi-encoder retriever, a learned stage-1 candidate filter (4.77 pairs / Source-1 entity), and cross-fitted
LightGBM + cross-encoder stages that decide one owner per record. France (no training labels) is handled by
an open-set variant trained only on India + US labels.

Public leaderboard macro F0.5: **0.987133**.

## Repository layout
```
Documentation_template.md              solution write-up (problem analysis, methodology, results)
code/business_entity_resolution/       pipeline source code, requirements, and run instructions
output/                                 final submission files (matching_results.tsv, candidate_pairs.tsv) — gitignored, large
```

## Getting started
See [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) for environment
setup, the code map, and the exact run order for reproducing the final submission.

See [Documentation_template.md](Documentation_template.md) for the full solution write-up.
