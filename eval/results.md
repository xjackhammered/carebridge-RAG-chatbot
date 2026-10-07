| Model | Chunk | Lang | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| multilingual-e5-base | 600 | all | 36 | 0.75 | 0.89 | 0.94 | 0.83 |
| multilingual-e5-base | 600 | en | 25 | 0.80 | 0.92 | 0.96 | 0.87 |
| multilingual-e5-base | 600 | bn | 11 | 0.64 | 0.82 | 0.91 | 0.75 |

- **multilingual-e5-base / chunk 600**: answerable top-1 score min/avg = 0.777/0.856; out-of-scope top-1 max/avg = 0.797/0.760; best MIN_SCORE = **0.777** (98% of questions classified correctly). Missed: q07, q32.
