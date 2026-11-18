# Analysis: q2_grouped

split `grouped`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 14

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 10.04M | 1479 | 0.68 | 40,230 | 0.50 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.40, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 59.5 | 51.6 | 55.2 | 62.7 | 29.3 | 68.1 | 86.3 | 76.5 | 68.3 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 35432 | 811 | 246 |
| positive | 1027 | 1652 | 113 |
| negative | 525 | 146 | 278 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 90.1 | 91.7 | 90.9 | 2792 |
| negative | 74.2 | 70.4 | 72.3 | 949 |
| macro avg | 82.2 | 81.0 | 81.6 | 3741 |
| weighted avg | 86.1 | 86.3 | 86.2 | 3741 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 84.2 | 90.1 | 87.1 | 1932 |
| neutral | 55.2 | 44.9 | 49.5 | 708 |
| negative | 68.6 | 68.0 | 68.3 | 463 |
| macro avg | 69.3 | 67.7 | 68.3 | 3103 |
| weighted avg | 75.3 | 76.5 | 75.7 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 10 | 2727 | 68.8 | 61.6 | 65.1 |
| mid (50-499) | 25 | 817 | 45.5 | 36.6 | 33.6 |
| rare (1-49) | 39 | 197 | 23.0 | 17.2 | 16.3 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 500 | 477 | 67.1 | 62.9 | 51.2 | 56.4 | 69.1 |
| Earphones | 390 | 797 | 66.5 | 60.9 | 57.7 | 59.3 | 74.0 |
| ChargingAdapter | 339 | 227 | 57.4 | 53.5 | 46.7 | 49.9 | 66.5 |
| Mouse | 292 | 438 | 59.2 | 53.5 | 53.9 | 53.7 | 51.2 |
| ChargingCable | 248 | 286 | 56.7 | 51.4 | 44.4 | 47.7 | 69.7 |
| Router | 223 | 259 | 53.2 | 54.6 | 41.3 | 47.0 | 54.1 |
| Keyboard | 221 | 214 | 67.5 | 66.7 | 57.0 | 61.5 | 63.9 |
| PowerBank | 148 | 153 | 57.6 | 55.9 | 43.1 | 48.7 | 66.3 |
| Microphone | 136 | 142 | 61.5 | 55.9 | 46.5 | 50.8 | 64.7 |
| Speaker | 129 | 159 | 66.7 | 67.4 | 54.7 | 60.4 | 60.0 |
| CellPhone | 106 | 157 | 74.6 | 68.8 | 67.5 | 68.2 | 52.4 |
| CC_Camera | 84 | 54 | 44.9 | 54.3 | 35.2 | 42.7 | 63.9 |
| Mini_Printer | 67 | 61 | 59.4 | 70.0 | 45.9 | 55.4 | 70.6 |
| SmartWatch | 60 | 92 | 53.8 | 53.1 | 37.0 | 43.6 | 75.9 |
| Pendrive | 57 | 70 | 49.6 | 51.0 | 37.1 | 43.0 | 73.1 |
| HDMI_Cable | 40 | 63 | 76.0 | 75.9 | 69.8 | 72.7 | 45.7 |
| MemoryCards | 23 | 36 | 64.8 | 60.0 | 58.3 | 59.2 | 58.2 |
| SmartPhone | 21 | 17 | 59.3 | 70.0 | 41.2 | 51.9 | 79.4 |
| WebCam | 19 | 39 | 80.0 | 92.3 | 61.5 | 73.8 | 57.0 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1380 | 1567 | 63.9 | 61.4 | 54.3 | 57.7 | 66.4 |
| bangla | 1160 | 1509 | 61.4 | 57.7 | 49.9 | 53.5 | 71.9 |
| banglish | 371 | 385 | 60.1 | 57.9 | 43.6 | 49.8 | 63.4 |
| mixed | 187 | 279 | 65.4 | 59.2 | 56.3 | 57.7 | 64.9 |
| other | 5 | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| price | 2265 | 487 | 75.5 | 69.6 | 91.0 |
| quality | 2299 | 472 | 64.7 | 61.6 | 92.8 |
| authenticity | 1495 | 320 | 67.0 | 61.2 | 90.0 |
| sound | 1513 | 309 | 80.5 | 66.1 | 83.2 |
| performance | 1323 | 296 | 74.1 | 66.8 | 89.5 |
| delivery | 1159 | 257 | 63.0 | 53.4 | 85.2 |
| seller | 1006 | 202 | 62.5 | 61.5 | 96.0 |
| battery | 637 | 141 | 79.3 | 61.8 | 78.7 |
| packaging | 691 | 129 | 49.8 | 42.9 | 82.2 |
| compatibility | 490 | 115 | 52.7 | 44.5 | 81.7 |
| build | 576 | 114 | 34.7 | 29.5 | 78.1 |
| speed | 442 | 114 | 70.3 | 55.9 | 81.6 |
| durability | 302 | 65 | 25.6 | 18.6 | 81.5 |
| connectivity | 162 | 51 | 7.0 | 7.0 | 78.4 |
| design | 243 | 50 | 16.9 | 16.9 | 88.0 |

## Per-aspect thresholds (tuned on validation, applied to test)

14 aspects with at least 50 validation mentions get their own threshold; the rest keep the global one (0.40).

| threshold | pair P | pair R | pair F1 | det F1 | det macro-F1 |
|---|---|---|---|---|---|
| global | 59.5 | 51.6 | 55.2 | 62.7 | 29.3 |
| per-aspect | 53.5 | 55.2 | 54.4 | 61.7 | 29.6 |
