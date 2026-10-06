| Model | Chunk | Lang | n | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|---|
| multilingual-e5-base | 300 | all | 30 | 0.70 | 0.93 | 0.97 | 0.82 |
| multilingual-e5-base | 300 | en | 21 | 0.76 | 1.00 | 1.00 | 0.88 |
| multilingual-e5-base | 300 | bn | 9 | 0.56 | 0.78 | 0.89 | 0.69 |
| multilingual-e5-base | 600 | all | 30 | 0.70 | 0.93 | 0.97 | 0.82 |
| multilingual-e5-base | 600 | en | 21 | 0.76 | 1.00 | 1.00 | 0.88 |
| multilingual-e5-base | 600 | bn | 9 | 0.56 | 0.78 | 0.89 | 0.69 |
| multilingual-e5-base | 1000 | all | 30 | 0.73 | 0.97 | 1.00 | 0.86 |
| multilingual-e5-base | 1000 | en | 21 | 0.76 | 1.00 | 1.00 | 0.88 |
| multilingual-e5-base | 1000 | bn | 9 | 0.67 | 0.89 | 1.00 | 0.81 |
| all-MiniLM-L6-v2 | 300 | all | 30 | 0.70 | 0.83 | 0.83 | 0.76 |
| all-MiniLM-L6-v2 | 300 | en | 21 | 0.86 | 1.00 | 1.00 | 0.93 |
| all-MiniLM-L6-v2 | 300 | bn | 9 | 0.33 | 0.44 | 0.44 | 0.37 |
| all-MiniLM-L6-v2 | 600 | all | 30 | 0.73 | 0.83 | 0.83 | 0.78 |
| all-MiniLM-L6-v2 | 600 | en | 21 | 0.90 | 1.00 | 1.00 | 0.95 |
| all-MiniLM-L6-v2 | 600 | bn | 9 | 0.33 | 0.44 | 0.44 | 0.37 |
| all-MiniLM-L6-v2 | 1000 | all | 30 | 0.70 | 0.83 | 0.83 | 0.76 |
| all-MiniLM-L6-v2 | 1000 | en | 21 | 0.86 | 1.00 | 1.00 | 0.92 |
| all-MiniLM-L6-v2 | 1000 | bn | 9 | 0.33 | 0.44 | 0.44 | 0.37 |

- **multilingual-e5-base / chunk 300**: answerable top-1 score min/avg = 0.789/0.865; out-of-scope top-1 max/avg = 0.797/0.760; best MIN_SCORE = **0.820** (97% of questions classified correctly). Missed: q07.
- **multilingual-e5-base / chunk 600**: answerable top-1 score min/avg = 0.786/0.864; out-of-scope top-1 max/avg = 0.797/0.760; best MIN_SCORE = **0.820** (97% of questions classified correctly). Missed: q07.
- **multilingual-e5-base / chunk 1000**: answerable top-1 score min/avg = 0.798/0.865; out-of-scope top-1 max/avg = 0.797/0.760; best MIN_SCORE = **0.798** (100% of questions classified correctly).
- **all-MiniLM-L6-v2 / chunk 300**: answerable top-1 score min/avg = 0.395/0.695; out-of-scope top-1 max/avg = 0.777/0.361; best MIN_SCORE = **0.395** (97% of questions classified correctly). Missed: q02, q04, q07, q11, q19.
- **all-MiniLM-L6-v2 / chunk 600**: answerable top-1 score min/avg = 0.395/0.693; out-of-scope top-1 max/avg = 0.777/0.356; best MIN_SCORE = **0.395** (97% of questions classified correctly). Missed: q02, q04, q07, q11, q19.
- **all-MiniLM-L6-v2 / chunk 1000**: answerable top-1 score min/avg = 0.395/0.691; out-of-scope top-1 max/avg = 0.777/0.355; best MIN_SCORE = **0.395** (97% of questions classified correctly). Missed: q02, q04, q07, q11, q19.
