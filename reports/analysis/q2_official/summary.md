# Analysis: q2_official

split `official`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 14

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 10.04M | 921 | 1.09 | 40,184 | 0.48 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.35, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 59.3 | 57.5 | 58.4 | 65.3 | 33.5 | 71.1 | 86.9 | 78.5 | 69.8 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 35178 | 896 | 336 |
| positive | 879 | 1789 | 106 |
| negative | 468 | 150 | 382 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 90.5 | 91.9 | 91.2 | 2774 |
| negative | 76.5 | 73.1 | 74.8 | 1000 |
| macro avg | 83.5 | 82.5 | 83.0 | 3774 |
| weighted avg | 86.8 | 86.9 | 86.8 | 3774 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 88.0 | 90.3 | 89.1 | 1983 |
| neutral | 53.5 | 54.5 | 54.0 | 653 |
| negative | 71.7 | 61.9 | 66.4 | 467 |
| macro avg | 71.1 | 68.9 | 69.8 | 3103 |
| weighted avg | 78.3 | 78.5 | 78.3 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 11 | 2896 | 70.5 | 63.5 | 66.8 |
| mid (50-499) | 25 | 709 | 47.9 | 41.5 | 38.8 |
| rare (1-49) | 37 | 167 | 27.6 | 20.9 | 19.5 |
| unseen (0) | 1 | 2 | 0.0 | 0.0 | 0.0 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 480 | 481 | 67.2 | 58.8 | 55.5 | 57.1 | 69.3 |
| Earphones | 392 | 810 | 69.7 | 60.6 | 64.3 | 62.4 | 68.2 |
| ChargingAdapter | 364 | 237 | 57.6 | 49.8 | 46.4 | 48.0 | 70.4 |
| Mouse | 310 | 452 | 62.8 | 58.2 | 60.6 | 59.4 | 62.1 |
| ChargingCable | 239 | 227 | 60.7 | 53.0 | 54.6 | 53.8 | 70.9 |
| Router | 230 | 290 | 61.3 | 56.7 | 51.0 | 53.7 | 68.9 |
| Keyboard | 222 | 212 | 65.8 | 67.2 | 58.0 | 62.3 | 64.5 |
| Microphone | 135 | 156 | 68.6 | 58.0 | 55.8 | 56.9 | 67.9 |
| Speaker | 123 | 172 | 70.1 | 66.0 | 62.2 | 64.1 | 70.2 |
| CellPhone | 120 | 179 | 79.3 | 75.3 | 73.2 | 74.2 | 68.1 |
| PowerBank | 110 | 114 | 54.1 | 53.8 | 43.9 | 48.3 | 70.2 |
| CC_Camera | 84 | 62 | 47.4 | 48.1 | 40.3 | 43.9 | 66.6 |
| Pendrive | 69 | 93 | 60.6 | 58.5 | 51.6 | 54.9 | 63.1 |
| Mini_Printer | 66 | 54 | 48.4 | 56.8 | 38.9 | 46.2 | 81.2 |
| SmartWatch | 65 | 83 | 50.6 | 41.8 | 39.8 | 40.7 | 60.4 |
| HDMI_Cable | 37 | 64 | 79.4 | 73.1 | 76.6 | 74.8 | 55.0 |
| SmartPhone | 22 | 19 | 75.7 | 72.2 | 68.4 | 70.3 | 77.6 |
| MemoryCards | 20 | 39 | 55.6 | 57.6 | 48.7 | 52.8 | 64.3 |
| WebCam | 15 | 30 | 79.3 | 75.0 | 70.0 | 72.4 | 52.5 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1416 | 1542 | 68.0 | 61.6 | 62.4 | 62.0 | 69.8 |
| bangla | 1137 | 1532 | 63.9 | 58.2 | 54.4 | 56.2 | 71.2 |
| banglish | 356 | 368 | 58.8 | 56.4 | 50.0 | 53.0 | 64.5 |
| mixed | 189 | 332 | 65.8 | 56.7 | 57.5 | 57.1 | 69.7 |
| other | 5 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 82.2 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| quality | 2312 | 501 | 73.3 | 70.8 | 94.4 |
| price | 2280 | 483 | 76.7 | 69.5 | 89.6 |
| authenticity | 1494 | 325 | 64.4 | 59.9 | 91.7 |
| sound | 1526 | 299 | 84.5 | 71.4 | 83.9 |
| performance | 1355 | 297 | 74.8 | 68.3 | 90.6 |
| delivery | 1165 | 255 | 66.9 | 53.7 | 80.4 |
| seller | 988 | 224 | 63.5 | 60.4 | 91.5 |
| packaging | 667 | 154 | 56.1 | 48.4 | 84.4 |
| battery | 633 | 134 | 76.7 | 64.7 | 83.6 |
| build | 551 | 121 | 41.3 | 34.7 | 79.3 |
| speed | 437 | 110 | 73.1 | 59.4 | 80.9 |
| compatibility | 500 | 103 | 56.5 | 51.1 | 90.3 |
| keys | 262 | 53 | 76.8 | 72.7 | 96.2 |
| durability | 344 | 50 | 38.1 | 35.7 | 94.0 |
| range | 157 | 46 | 61.5 | 48.4 | 71.7 |

## Per-aspect thresholds (tuned on validation, applied to test)

15 aspects with at least 50 validation mentions get their own threshold; the rest keep the global one (0.35).

| threshold | pair P | pair R | pair F1 | det F1 | det macro-F1 |
|---|---|---|---|---|---|
| global | 59.3 | 57.5 | 58.4 | 65.3 | 33.5 |
| per-aspect | 58.0 | 57.0 | 57.5 | 64.4 | 33.6 |
