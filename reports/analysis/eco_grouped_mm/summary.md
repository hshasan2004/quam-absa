# Analysis: eco_grouped_mm

split `grouped`, layout `mm`, readout `memory`, presence `mag+value`, seed 42, best epoch 15

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 5.78M | 760 | 1.32 | 40,230 | 0.47 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.30, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 45.9 | 44.8 | 45.4 | 50.1 | 16.7 | 52.9 | 86.1 | 77.2 | 69.2 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 34692 | 1430 | 367 |
| positive | 1253 | 1472 | 67 |
| negative | 635 | 110 | 204 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 90.4 | 91.1 | 90.7 | 2792 |
| negative | 73.1 | 71.4 | 72.3 | 949 |
| macro avg | 81.8 | 81.3 | 81.5 | 3741 |
| weighted avg | 86.0 | 86.1 | 86.0 | 3741 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 85.7 | 89.8 | 87.7 | 1932 |
| neutral | 56.2 | 50.8 | 53.4 | 708 |
| negative | 68.6 | 64.8 | 66.7 | 463 |
| macro avg | 70.2 | 68.5 | 69.2 | 3103 |
| weighted avg | 76.4 | 77.2 | 76.7 | 3103 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 10 | 2727 | 55.7 | 50.9 | 44.7 |
| mid (50-499) | 25 | 817 | 32.5 | 26.9 | 20.9 |
| rare (1-49) | 39 | 197 | 5.5 | 4.6 | 6.0 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Earbuds | 500 | 477 | 45.4 | 47.3 | 35.2 | 40.4 | 69.4 |
| Earphones | 390 | 797 | 55.5 | 45.0 | 55.0 | 49.5 | 74.9 |
| ChargingAdapter | 339 | 227 | 41.9 | 39.6 | 36.1 | 37.8 | 67.5 |
| Mouse | 292 | 438 | 50.0 | 43.7 | 47.7 | 45.6 | 56.2 |
| ChargingCable | 248 | 286 | 43.1 | 41.9 | 36.4 | 39.0 | 68.2 |
| Router | 223 | 259 | 40.9 | 44.9 | 34.0 | 38.7 | 52.0 |
| Keyboard | 221 | 214 | 52.2 | 48.3 | 47.7 | 48.0 | 67.8 |
| PowerBank | 148 | 153 | 45.4 | 42.0 | 37.9 | 39.9 | 64.1 |
| Microphone | 136 | 142 | 51.0 | 49.6 | 40.8 | 44.8 | 70.9 |
| Speaker | 129 | 159 | 57.8 | 52.4 | 54.7 | 53.5 | 60.2 |
| CellPhone | 106 | 157 | 63.4 | 57.6 | 60.5 | 59.0 | 65.1 |
| CC_Camera | 84 | 54 | 32.7 | 30.0 | 27.8 | 28.8 | 63.7 |
| Mini_Printer | 67 | 61 | 42.4 | 50.0 | 31.1 | 38.4 | 57.7 |
| SmartWatch | 60 | 92 | 34.4 | 31.8 | 30.4 | 31.1 | 78.4 |
| Pendrive | 57 | 70 | 43.5 | 44.4 | 34.3 | 38.7 | 66.6 |
| HDMI_Cable | 40 | 63 | 71.3 | 65.2 | 68.3 | 66.7 | 53.5 |
| MemoryCards | 23 | 36 | 60.2 | 48.9 | 63.9 | 55.4 | 78.5 |
| SmartPhone | 21 | 17 | 43.7 | 40.0 | 35.3 | 37.5 | 82.8 |
| WebCam | 19 | 39 | 83.1 | 76.3 | 74.4 | 75.3 | 65.2 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1380 | 1567 | 51.2 | 47.9 | 46.8 | 47.4 | 68.5 |
| bangla | 1160 | 1509 | 49.4 | 45.6 | 42.6 | 44.0 | 72.3 |
| banglish | 371 | 385 | 47.5 | 42.3 | 42.3 | 42.3 | 61.4 |
| mixed | 187 | 279 | 51.9 | 42.5 | 48.4 | 45.2 | 66.3 |
| other | 5 | 1 | 50.0 | 33.3 | 100.0 | 50.0 | 100.0 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| price | 2265 | 487 | 76.2 | 71.3 | 90.6 |
| quality | 2299 | 472 | 58.7 | 54.5 | 91.5 |
| authenticity | 1495 | 320 | 64.2 | 60.7 | 91.2 |
| sound | 1513 | 309 | 57.6 | 47.8 | 82.5 |
| performance | 1323 | 296 | 64.9 | 58.7 | 90.2 |
| delivery | 1159 | 257 | 22.4 | 20.8 | 87.5 |
| seller | 1006 | 202 | 43.8 | 43.4 | 94.6 |
| battery | 637 | 141 | 26.3 | 20.3 | 80.1 |
| packaging | 691 | 129 | 9.3 | 9.3 | 82.9 |
| compatibility | 490 | 115 | 25.5 | 22.9 | 87.0 |
| build | 576 | 114 | 23.6 | 18.2 | 75.4 |
| speed | 442 | 114 | 57.0 | 47.7 | 82.5 |
| durability | 302 | 65 | 8.8 | 4.4 | 78.5 |
| connectivity | 162 | 51 | 0.0 | 0.0 | 82.4 |
| design | 243 | 50 | 17.4 | 17.4 | 84.0 |
