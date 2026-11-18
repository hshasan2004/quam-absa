# Analysis: q2_zeroshot

split `zeroshot`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 14

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 10.04M | 1310 | 0.76 | 32,104 | 0.71 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.30, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 40.1 | 43.3 | 41.6 | 47.2 | 29.1 | 49.5 | 86.1 | 73.6 | 61.7 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 27737 | 1272 | 345 |
| positive | 989 | 1051 | 61 |
| negative | 411 | 99 | 139 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 89.7 | 92.5 | 91.0 | 2101 |
| negative | 72.9 | 65.5 | 69.0 | 649 |
| macro avg | 81.3 | 79.0 | 80.0 | 2750 |
| weighted avg | 85.7 | 86.1 | 85.8 | 2750 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 81.1 | 89.7 | 85.2 | 1756 |
| neutral | 46.6 | 33.9 | 39.2 | 614 |
| negative | 62.8 | 58.8 | 60.7 | 267 |
| macro avg | 63.5 | 60.8 | 61.7 | 2637 |
| weighted avg | 71.3 | 73.6 | 72.0 | 2637 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 8 | 1464 | 69.0 | 62.9 | 61.0 |
| mid (50-499) | 5 | 260 | 38.9 | 29.5 | 30.6 |
| rare (1-49) | 2 | 70 | 2.7 | 2.7 | 3.7 |
| unseen (0) | 18 | 956 | 21.6 | 17.6 | 17.3 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Router | 1510 | 1728 | 51.5 | 47.9 | 43.4 | 45.6 | 54.1 |
| CC_Camera | 554 | 400 | 36.6 | 23.2 | 52.0 | 32.1 | 69.1 |
| Mini_Printer | 449 | 387 | 30.9 | 32.7 | 22.0 | 26.3 | 68.9 |
| WebCam | 124 | 235 | 68.6 | 59.8 | 62.6 | 61.1 | 65.1 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1130 | 1082 | 51.1 | 48.8 | 45.8 | 47.3 | 62.1 |
| bangla | 1050 | 1135 | 44.5 | 34.3 | 43.3 | 38.3 | 62.2 |
| banglish | 245 | 202 | 37.2 | 34.4 | 30.7 | 32.5 | 53.1 |
| mixed | 205 | 331 | 51.4 | 41.8 | 42.3 | 42.0 | 62.6 |
| other | 7 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 77.8 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| price | 2715 | 471 | 77.2 | 70.2 | 91.1 |
| delivery | 1322 | 287 | 76.7 | 68.6 | 86.4 |
| range | 0 | 237 | 0.0 | 0.0 | 81.9 |
| packaging | 702 | 220 | 53.1 | 47.8 | 86.8 |
| seller | 1143 | 201 | 67.6 | 63.9 | 92.5 |
| speed | 496 | 144 | 40.4 | 31.6 | 79.2 |
| authenticity | 1913 | 142 | 55.7 | 53.7 | 93.7 |
| print | 0 | 138 | 4.2 | 4.2 | 91.3 |
| stability | 0 | 137 | 17.4 | 10.3 | 81.0 |
| quality | 3201 | 111 | 85.7 | 76.2 | 90.1 |
| durability and quality | 0 | 92 | 30.7 | 27.5 | 90.2 |
| firmware | 0 | 62 | 23.1 | 21.5 | 83.9 |
| video quality | 0 | 60 | 38.0 | 31.3 | 83.3 |
| warranty | 110 | 46 | 44.1 | 26.5 | 63.0 |
| setup | 8 | 45 | 0.0 | 0.0 | 91.1 |

## Per-aspect thresholds (tuned on validation, applied to test)

7 aspects with at least 50 validation mentions get their own threshold; the rest keep the global one (0.30).

| threshold | pair P | pair R | pair F1 | det F1 | det macro-F1 |
|---|---|---|---|---|---|
| global | 40.1 | 43.3 | 41.6 | 47.2 | 29.1 |
| per-aspect | 39.9 | 41.3 | 40.6 | 46.2 | 29.3 |
