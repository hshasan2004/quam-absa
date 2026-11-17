# QuAM-xLSTM: Query-the-Aspect-Memory

Aspect-based sentiment analysis for Bangla, Banglish and English reviews. A bidirectional
xLSTM reads a review once. Each aspect is then answered by querying the final mLSTM matrix
memory with the aspect name:

- **presence** from the memory normaliser, `log(|n_T^T q| + 1)`: is the aspect mentioned?
- **sentiment** from the retrieved value, `C_T q / max(|n_T^T q|, 1)`: positive, neutral or negative?

Overall review sentiment is a learned, presence-weighted aggregation of the aspect sentiments.

## Datasets (not included; download them yourself)
- **EComReviews-BD**: https://github.com/Afarrobo/PRODUCT_REVIEWS → `data/raw/ecomreviews/`
- **BABSA**: https://data.mendeley.com/datasets/j7yb2sv263/1 (CC BY-NC-SA 4.0) → `data/raw/babsa_mendeley/`

## Pipeline
```
python scripts/eda.py          # dataset analysis -> reports/eda_report.txt
python scripts/preprocess.py   # unified JSONL + splits -> data/processed/
python scripts/encode.py       # frozen XLM-R features (memory-mapped cache)
python tests/test_xlstm.py     # numerical checks (incl. vs. official NX-AI xlstm)
python scripts/train.py --name eco_official --data eco --eco-split official
python scripts/run_queue.py    # all main experiments + ablations
python scripts/summarize.py    # results table -> reports/results.md
```

## Notes
- XLM-R has outlier ("rogue") feature dimensions that make all aspect queries nearly parallel
  (cosine 0.989). Features are standardised per dimension before use.
- Gate pre-activations are soft-capped (`15 tanh(x/15)`, as in xLSTM 7B) to keep the
  stabiliser bounded and training finite.
