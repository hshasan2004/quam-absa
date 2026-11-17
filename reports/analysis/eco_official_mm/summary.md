# Analysis: eco_official_mm

split `official`, layout `mm`, readout `memory`, presence `mag+value`, seed 42, best epoch 14

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 5.78M | 659 | 1.52 | 40,184 | 0.45 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.30, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 51.9 | 52.5 | 52.2 | 57.9 | 20.8 | 61.4 | 86.5 | 80.7 | 74.0 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 34791 | 1236 | 383 |
| positive | 983 | 1700 | 91 |
| negative | 596 | 123 | 281 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 90.6 | 91.1 | 90.8 | 2774 |
| negative | 74.8 | 73.7 | 74.3 | 1000 |
| macro avg | 82.7 | 82.4 | 82.5 | 3774 |
| weighted avg | 86.4 | 86.5 | 86.4 | 3774 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 88.8 | 89.7 | 89.3 | 1983 |
| neutral | 59.2 | 62.3 | 60.7 | 653 |
| negative | 76.9 | 67.9 | 72.1 | 467 |
| macro avg | 75.0 | 73.3 | 74.0 | 3103 |
| weighted avg | 80.8 | 80.7 | 80.7 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 11 | 2896 | 63.8 | 58.0 | 55.8 |
| mid (50-499) | 25 | 709 | 34.5 | 28.5 | 21.9 |
| rare (1-49) | 37 | 167 | 10.8 | 9.8 | 9.0 |
| unseen (0) | 1 | 2 | 0.0 | 0.0 | 0.0 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 480 | 481 | 53.2 | 46.7 | 47.8 | 47.2 | 74.7 |
| Earphones | 392 | 810 | 65.0 | 55.4 | 60.0 | 57.6 | 72.8 |
| ChargingAdapter | 364 | 237 | 52.9 | 49.1 | 46.4 | 47.7 | 73.8 |
| Mouse | 310 | 452 | 58.2 | 52.9 | 57.3 | 55.0 | 67.9 |
| ChargingCable | 239 | 227 | 51.0 | 43.3 | 45.8 | 44.5 | 73.0 |
| Router | 230 | 290 | 50.0 | 44.5 | 42.1 | 43.3 | 71.5 |
| Keyboard | 222 | 212 | 53.8 | 51.8 | 48.1 | 49.9 | 74.0 |
| Microphone | 135 | 156 | 62.1 | 52.9 | 51.9 | 52.4 | 70.8 |
| Speaker | 123 | 172 | 62.3 | 56.2 | 58.1 | 57.1 | 75.1 |
| CellPhone | 120 | 179 | 74.1 | 70.6 | 70.9 | 70.8 | 66.8 |
| PowerBank | 110 | 114 | 51.7 | 56.3 | 43.0 | 48.8 | 73.9 |
| CC_Camera | 84 | 62 | 43.9 | 42.6 | 41.9 | 42.3 | 63.9 |
| Pendrive | 69 | 93 | 51.2 | 49.3 | 37.6 | 42.7 | 71.7 |
| Mini_Printer | 66 | 54 | 44.2 | 42.0 | 38.9 | 40.4 | 81.8 |
| SmartWatch | 65 | 83 | 39.5 | 34.8 | 37.3 | 36.0 | 73.8 |
| HDMI_Cable | 37 | 64 | 78.5 | 70.4 | 78.1 | 74.1 | 61.6 |
| SmartPhone | 22 | 19 | 60.0 | 52.4 | 57.9 | 55.0 | 92.3 |
| MemoryCards | 20 | 39 | 53.5 | 50.0 | 41.0 | 45.1 | 84.9 |
| WebCam | 15 | 30 | 78.6 | 80.8 | 70.0 | 75.0 | 100.0 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1416 | 1542 | 58.7 | 53.5 | 55.4 | 54.4 | 73.8 |
| bangla | 1137 | 1532 | 57.6 | 50.8 | 51.4 | 51.1 | 75.7 |
| banglish | 356 | 368 | 55.2 | 52.6 | 47.3 | 49.8 | 71.8 |
| mixed | 189 | 332 | 58.3 | 49.0 | 49.7 | 49.3 | 68.3 |
| other | 5 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 61.9 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| quality | 2312 | 501 | 70.2 | 67.6 | 93.2 |
| price | 2280 | 483 | 78.2 | 71.0 | 88.8 |
| authenticity | 1494 | 325 | 64.6 | 61.2 | 92.0 |
| sound | 1526 | 299 | 68.4 | 60.6 | 86.3 |
| performance | 1355 | 297 | 70.5 | 60.7 | 86.5 |
| delivery | 1165 | 255 | 41.9 | 35.2 | 82.4 |
| seller | 988 | 224 | 68.1 | 64.2 | 89.7 |
| packaging | 667 | 154 | 24.0 | 19.2 | 85.7 |
| battery | 633 | 134 | 42.5 | 37.8 | 85.1 |
| build | 551 | 121 | 35.2 | 29.5 | 75.2 |
| speed | 437 | 110 | 61.3 | 47.7 | 79.1 |
| compatibility | 500 | 103 | 50.2 | 45.7 | 85.4 |
| keys | 262 | 53 | 54.2 | 47.5 | 88.7 |
| durability | 344 | 50 | 14.6 | 14.6 | 86.0 |
| range | 157 | 46 | 49.5 | 40.9 | 80.4 |
