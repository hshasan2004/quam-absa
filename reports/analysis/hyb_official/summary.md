# Analysis: hyb_official

split `official`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 14

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 7.23M | 1130 | 0.89 | 40,184 | 0.46 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.25, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 41.5 | 50.7 | 45.6 | 51.3 | 19.1 | 51.4 | 86.4 | 78.9 | 70.1 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 33948 | 1918 | 544 |
| positive | 1041 | 1631 | 102 |
| negative | 583 | 136 | 281 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 90.3 | 91.3 | 90.8 | 2774 |
| negative | 75.0 | 72.7 | 73.8 | 1000 |
| macro avg | 82.6 | 82.0 | 82.3 | 3774 |
| weighted avg | 86.2 | 86.4 | 86.3 | 3774 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 88.1 | 90.7 | 89.4 | 1983 |
| neutral | 55.6 | 54.8 | 55.2 | 653 |
| negative | 69.7 | 62.1 | 65.7 | 467 |
| macro avg | 71.1 | 69.2 | 70.1 | 3103 |
| weighted avg | 78.5 | 78.9 | 78.6 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 11 | 2896 | 55.4 | 49.4 | 48.9 |
| mid (50-499) | 25 | 709 | 36.5 | 31.4 | 22.7 |
| rare (1-49) | 37 | 167 | 9.6 | 9.6 | 7.0 |
| unseen (0) | 1 | 2 | 0.0 | 0.0 | 0.0 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 480 | 481 | 46.9 | 37.8 | 44.3 | 40.8 | 70.2 |
| Earphones | 392 | 810 | 59.6 | 48.6 | 59.0 | 53.3 | 70.0 |
| ChargingAdapter | 364 | 237 | 45.7 | 33.2 | 43.5 | 37.7 | 71.2 |
| Mouse | 310 | 452 | 52.0 | 42.4 | 52.9 | 47.0 | 62.2 |
| ChargingCable | 239 | 227 | 43.7 | 32.9 | 46.7 | 38.6 | 67.1 |
| Router | 230 | 290 | 43.9 | 33.5 | 43.8 | 38.0 | 63.6 |
| Keyboard | 222 | 212 | 48.1 | 43.5 | 48.6 | 45.9 | 65.3 |
| Microphone | 135 | 156 | 54.0 | 40.5 | 48.1 | 44.0 | 71.2 |
| Speaker | 123 | 172 | 56.8 | 46.4 | 52.3 | 49.2 | 71.0 |
| CellPhone | 120 | 179 | 69.0 | 68.6 | 64.8 | 66.7 | 65.2 |
| PowerBank | 110 | 114 | 44.7 | 39.8 | 43.0 | 41.4 | 69.5 |
| CC_Camera | 84 | 62 | 37.5 | 30.5 | 40.3 | 34.7 | 67.3 |
| Pendrive | 69 | 93 | 47.3 | 33.6 | 47.3 | 39.3 | 79.0 |
| Mini_Printer | 66 | 54 | 39.6 | 36.5 | 35.2 | 35.8 | 62.1 |
| SmartWatch | 65 | 83 | 34.2 | 24.1 | 42.2 | 30.7 | 66.2 |
| HDMI_Cable | 37 | 64 | 64.7 | 53.3 | 62.5 | 57.6 | 65.0 |
| SmartPhone | 22 | 19 | 39.1 | 33.3 | 47.4 | 39.1 | 87.9 |
| MemoryCards | 20 | 39 | 55.7 | 50.0 | 51.3 | 50.6 | 73.3 |
| WebCam | 15 | 30 | 71.2 | 72.4 | 70.0 | 71.2 | 84.1 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1416 | 1542 | 51.7 | 43.2 | 53.6 | 47.8 | 70.3 |
| bangla | 1137 | 1532 | 51.2 | 40.5 | 49.7 | 44.6 | 70.6 |
| banglish | 356 | 368 | 48.7 | 40.4 | 44.8 | 42.5 | 67.3 |
| mixed | 189 | 332 | 52.0 | 39.2 | 47.9 | 43.1 | 71.4 |
| other | 5 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 33.3 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| quality | 2312 | 501 | 66.1 | 62.1 | 93.0 |
| price | 2280 | 483 | 61.4 | 55.7 | 89.0 |
| authenticity | 1494 | 325 | 50.1 | 45.8 | 90.5 |
| sound | 1526 | 299 | 62.8 | 51.9 | 83.9 |
| performance | 1355 | 297 | 71.3 | 63.0 | 89.2 |
| delivery | 1165 | 255 | 34.6 | 27.9 | 81.6 |
| seller | 988 | 224 | 48.4 | 46.1 | 90.6 |
| packaging | 667 | 154 | 25.3 | 24.5 | 88.3 |
| battery | 633 | 134 | 40.1 | 34.4 | 83.6 |
| build | 551 | 121 | 31.9 | 23.7 | 73.6 |
| speed | 437 | 110 | 55.2 | 44.0 | 80.0 |
| compatibility | 500 | 103 | 45.5 | 37.8 | 85.4 |
| keys | 262 | 53 | 53.6 | 51.0 | 92.5 |
| durability | 344 | 50 | 12.5 | 12.5 | 94.0 |
| range | 157 | 46 | 54.7 | 43.4 | 78.3 |

## Per-aspect thresholds (tuned on validation, applied to test)

15 aspects with at least 50 validation mentions get their own threshold; the rest keep the global one (0.25).

| threshold | pair P | pair R | pair F1 | det F1 | det macro-F1 |
|---|---|---|---|---|---|
| global | 41.5 | 50.7 | 45.6 | 51.3 | 19.1 |
| per-aspect | 38.3 | 48.2 | 42.7 | 47.8 | 19.2 |
