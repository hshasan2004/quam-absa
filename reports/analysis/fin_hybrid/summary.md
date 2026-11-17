# Analysis: fin_hybrid

split `grouped`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 15

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 7.23M | 1828 | 0.55 | 40,230 | 0.48 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.25, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 40.7 | 50.1 | 44.9 | 50.9 | 19.8 | 50.1 | 86.0 | 76.3 | 68.5 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 34006 | 2044 | 439 |
| positive | 1063 | 1647 | 82 |
| negative | 556 | 166 | 227 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 89.2 | 92.4 | 90.8 | 2792 |
| negative | 74.9 | 67.1 | 70.8 | 949 |
| macro avg | 82.1 | 79.7 | 80.8 | 3741 |
| weighted avg | 85.6 | 86.0 | 85.7 | 3741 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 84.6 | 89.1 | 86.8 | 1932 |
| neutral | 54.1 | 48.4 | 51.1 | 708 |
| negative | 69.8 | 65.4 | 67.6 | 463 |
| macro avg | 69.5 | 67.7 | 68.5 | 3103 |
| weighted avg | 75.4 | 76.3 | 75.8 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 10 | 2727 | 55.1 | 49.4 | 46.2 |
| mid (50-499) | 25 | 817 | 38.1 | 30.6 | 22.7 |
| rare (1-49) | 39 | 197 | 16.5 | 14.8 | 10.5 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 500 | 477 | 49.8 | 45.8 | 41.3 | 43.4 | 68.4 |
| Earphones | 390 | 797 | 57.0 | 41.2 | 59.8 | 48.8 | 71.4 |
| ChargingAdapter | 339 | 227 | 39.8 | 32.6 | 33.0 | 32.8 | 64.6 |
| Mouse | 292 | 438 | 47.9 | 35.0 | 59.4 | 44.0 | 54.1 |
| ChargingCable | 248 | 286 | 45.0 | 32.4 | 46.2 | 38.1 | 66.6 |
| Router | 223 | 259 | 43.4 | 40.7 | 39.0 | 39.8 | 47.9 |
| Keyboard | 221 | 214 | 55.0 | 48.7 | 51.4 | 50.0 | 72.1 |
| PowerBank | 148 | 153 | 38.8 | 28.6 | 43.8 | 34.6 | 69.4 |
| Microphone | 136 | 142 | 47.9 | 43.7 | 43.7 | 43.7 | 73.2 |
| Speaker | 129 | 159 | 63.0 | 57.6 | 59.7 | 58.6 | 68.2 |
| CellPhone | 106 | 157 | 64.7 | 58.7 | 58.0 | 58.3 | 66.0 |
| CC_Camera | 84 | 54 | 35.9 | 28.6 | 33.3 | 30.8 | 61.2 |
| Mini_Printer | 67 | 61 | 53.2 | 49.2 | 50.8 | 50.0 | 58.2 |
| SmartWatch | 60 | 92 | 38.9 | 31.6 | 33.7 | 32.6 | 79.9 |
| Pendrive | 57 | 70 | 47.5 | 44.9 | 44.3 | 44.6 | 60.6 |
| HDMI_Cable | 40 | 63 | 69.6 | 56.0 | 66.7 | 60.9 | 58.9 |
| MemoryCards | 23 | 36 | 61.0 | 47.8 | 61.1 | 53.7 | 47.6 |
| SmartPhone | 21 | 17 | 43.2 | 35.0 | 41.2 | 37.8 | 82.8 |
| WebCam | 19 | 39 | 76.1 | 78.1 | 64.1 | 70.4 | 67.9 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1380 | 1567 | 52.0 | 42.8 | 53.3 | 47.4 | 68.7 |
| bangla | 1160 | 1509 | 50.1 | 39.0 | 47.6 | 42.9 | 70.9 |
| banglish | 371 | 385 | 48.2 | 40.0 | 43.1 | 41.5 | 59.7 |
| mixed | 187 | 279 | 51.1 | 39.2 | 54.8 | 45.7 | 67.5 |
| other | 5 | 1 | 66.7 | 50.0 | 100.0 | 66.7 | 80.0 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| price | 2265 | 487 | 78.6 | 71.2 | 89.5 |
| quality | 2299 | 472 | 55.0 | 50.1 | 91.3 |
| authenticity | 1495 | 320 | 44.2 | 40.9 | 90.3 |
| sound | 1513 | 309 | 67.8 | 57.1 | 85.1 |
| performance | 1323 | 296 | 65.2 | 57.6 | 88.9 |
| delivery | 1159 | 257 | 32.1 | 29.5 | 88.3 |
| seller | 1006 | 202 | 43.9 | 43.9 | 96.0 |
| battery | 637 | 141 | 33.4 | 25.2 | 76.6 |
| packaging | 691 | 129 | 11.8 | 11.8 | 83.7 |
| compatibility | 490 | 115 | 46.6 | 36.6 | 77.4 |
| build | 576 | 114 | 30.0 | 23.8 | 73.7 |
| speed | 442 | 114 | 55.5 | 46.2 | 86.8 |
| durability | 302 | 65 | 0.0 | 0.0 | 84.6 |
| connectivity | 162 | 51 | 0.0 | 0.0 | 70.6 |
| design | 243 | 50 | 12.7 | 12.7 | 88.0 |

## Per-aspect thresholds (tuned on validation, applied to test)

14 aspects with at least 50 validation mentions get their own threshold; the rest keep the global one (0.25).

| threshold | pair P | pair R | pair F1 | det F1 | det macro-F1 |
|---|---|---|---|---|---|
| global | 40.7 | 50.1 | 44.9 | 50.9 | 19.8 |
| per-aspect | 36.1 | 49.7 | 41.8 | 47.7 | 20.3 |
