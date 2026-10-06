| Model | Chunk | Lang | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| multilingual-e5-base | 600 | all | 30 | 0.83 | 0.93 | 0.97 | 0.89 |
| multilingual-e5-base | 600 | en | 21 | 0.90 | 1.00 | 1.00 | 0.95 |
| multilingual-e5-base | 600 | bn | 9 | 0.67 | 0.78 | 0.89 | 0.75 |

- **multilingual-e5-base / chunk 600**: answerable top-1 score min/avg = 0.795/0.865; out-of-scope top-1 max/avg = 0.797/0.760; best MIN_SCORE = **0.820** (97% of questions classified correctly). Missed: q07.
