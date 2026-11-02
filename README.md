# How small can retrieval vectors get before retrieval breaks

Code and saved results for a test of two ways to shrink the stored vectors of a search index: keep fewer numbers per vector (PCA), or store each number in fewer bits (FP16, int8, 1 bit). Three BEIR question sets (SciFact, NFCorpus, FiQA), two frozen encoders (BAAI bge-small-en-v1.5 and bge-base-en-v1.5), exact search in NumPy, nDCG@10.

## The pass mark, fixed before the first scored run

A setup passes only if all three hold, compared with the full FP32 vector:

1. Mean nDCG@10 falls by at most 1.0 point (0.01).
2. The upper end of the 95 percent paired bootstrap range of the loss is at most 2.0 points (10,000 resamples over questions).
3. At most 5 percent of questions lose 0.2 or more of their own nDCG@10.

## Smallest passing setup per test

Index size counts the vectors, the PCA matrix and the int8 scales.

| Dataset and encoder | Setup | Index size, percent of full | Mean loss, points | Upper end, points | Questions losing 0.2 or more, percent |
|---|---|---:|---:|---:|---:|
| SciFact, bge-small | 384 numbers, int8 | 25.0 | -0.2 | 0.0 | 0.0 |
| SciFact, bge-base | 768 numbers, int8 | 25.0 | 0.0 | 0.3 | 0.3 |
| NFCorpus, bge-small | 256 numbers, int8 | 23.7 | 0.3 | 0.7 | 0.0 |
| NFCorpus, bge-base | 256 numbers, int8 | 15.4 | 0.4 | 0.9 | 0.3 |
| FiQA, bge-small | 384 numbers, int8 | 25.0 | 0.0 | 0.2 | 0.2 |
| FiQA, bge-base | 512 numbers, int8 | 17.6 | 0.2 | 0.4 | 0.3 |

`results/grid.jsonl` has all 261 setups. `results/per_query_ndcg.npz` has the nDCG@10 of every question under every setup.

## Method notes

- PCA directions come from how a fixed random sample of 2,000 documents (seed 0) spreads around its average. No questions and no relevance labels are used. The rotation is applied to the vectors as they are, without subtracting the average, then each vector is scaled back to length one. A pilot on SciFact showed that subtracting the average (centring) cost 2.3 points even with all 384 numbers kept, so the full run does not centre. `run_grid.py` also reports the centred version once, as `trap_centred`.
- Documents are stored in the smaller format. Questions stay FP32. The int8 scale per dimension comes from the fitting sample only.
- Search is exact. Ties are broken by lower document index.
- The nDCG code in `lib.py` matched `pytrec_eval` exactly on the pilot.
- The pass mark was not changed after any result. Only the centring choice was changed, after the pilot and before the full run.

## Files

- `prepare_data.py` downloads the BEIR test splits from the BeIR repositories on Hugging Face into `data/`. The data and the saved embeddings are not included.
- `embed.py`, `embed_all.sh` embed each dataset with each encoder (about 85 seconds for SciFact with bge-small on an Apple silicon Mac, longer for FiQA with bge-base).
- `lib.py` loading, top ten search and nDCG@10. `compress.py` PCA, truncation and the precision formats.
- `run_grid.py` runs every setup and writes `results/grid.jsonl` and `results/per_query_ndcg.npz`.
- `analyze.py` prints the smallest passing setups. `per_query.py` prints the per question view. `make_chart.py` draws the chart in `figures/`.

## Running it

```
pip install -r requirements.txt
python prepare_data.py
./embed_all.sh
python run_grid.py
python analyze.py
python per_query.py
```

No API calls are made, so the cost is zero. Timing was not measured.

## Limits

Three English datasets, two encoders from one family, one PCA method, exact search, and a pass mark chosen by the author. The bootstrap resamples questions only. A different random fitting sample moved some low dimension scores by up to 1.6 points (see the `A_dims_seed` rows). Read about 12 hand picked and random questions, not all of them.

## Data and license notes

The BEIR datasets and the bge encoders belong to their authors. Check their terms before reuse. No license file is chosen for the code yet.
