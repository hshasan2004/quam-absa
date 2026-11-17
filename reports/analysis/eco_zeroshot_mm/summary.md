# Analysis: eco_zeroshot_mm

split `zeroshot`, layout `mm`, readout `memory`, presence `mag+value`, seed 42, best epoch 13

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 5.78M | 1558 | 0.64 | 32,104 | 0.68 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## EComReviews-BD (threshold 0.15, chosen on validation)

| pair P | pair R | pair F1 | det F1 | det macro-F1 | detection AP | polarity acc | overall acc | overall macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 20.1 | 52.6 | 29.0 | 33.2 | 24.7 | 28.5 | 86.0 | 71.7 | 59.5 |

Figures: cm_eco_joint, cm_eco_detection, cm_eco_polarity, cm_eco_overall, pr_eco_detection, threshold_sweep, aspect_f1_vs_freq (.png / .pdf)

### Joint confusion (rows gold, columns predicted)

| gold \ pred | absent | positive | negative |
|---|---|---|---|
| absent | 23795 | 4395 | 1164 |
| positive | 742 | 1291 | 68 |
| negative | 352 | 141 | 156 |

### Aspect polarity on gold-mentioned aspects

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 89.4 | 92.8 | 91.0 | 2101 |
| negative | 73.3 | 64.3 | 68.5 | 649 |
| macro avg | 81.3 | 78.5 | 79.8 | 2750 |
| weighted avg | 85.6 | 86.0 | 85.7 | 2750 |

### Overall review sentiment

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 80.6 | 87.6 | 84.0 | 1756 |
| neutral | 42.9 | 31.1 | 36.1 | 614 |
| negative | 56.9 | 60.3 | 58.5 | 267 |
| macro avg | 60.1 | 59.7 | 59.5 | 2637 |
| weighted avg | 69.4 | 71.7 | 70.3 | 2637 |

### By training frequency of the aspect

| bucket | aspects | test mentions | det F1 | pair F1 | det macro-F1 |
|---|---|---|---|---|---|
| frequent (500+) | 8 | 1464 | 41.9 | 38.2 | 40.4 |
| mid (50-499) | 5 | 260 | 21.1 | 15.7 | 12.8 |
| rare (1-49) | 2 | 70 | 15.2 | 15.2 | 15.7 |
| unseen (0) | 18 | 956 | 24.9 | 20.2 | 22.0 |

### By product category

| category | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| Router | 1510 | 1728 | 32.9 | 19.3 | 54.8 | 28.6 | 51.9 |
| CC_Camera | 554 | 400 | 29.2 | 17.3 | 58.8 | 26.7 | 70.2 |
| Mini_Printer | 449 | 387 | 26.3 | 19.4 | 26.4 | 22.3 | 63.7 |
| WebCam | 124 | 235 | 56.9 | 38.0 | 69.4 | 49.1 | 57.7 |

### By script

| script | reviews | mentions | det F1 | pair P | pair R | pair F1 | overall F1 |
|---|---|---|---|---|---|---|---|
| english | 1130 | 1082 | 33.5 | 22.4 | 51.0 | 31.2 | 62.9 |
| bangla | 1050 | 1135 | 32.6 | 18.3 | 55.9 | 27.6 | 59.0 |
| banglish | 245 | 202 | 27.8 | 17.1 | 36.6 | 23.3 | 42.8 |
| mixed | 205 | 331 | 37.9 | 21.9 | 56.2 | 31.5 | 60.0 |
| other | 7 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 41.3 |

### Per aspect: see per_aspect.csv. Top 15 by test mentions

| aspect | train freq | test mentions | det F1 | pair F1 | polarity acc |
|---|---|---|---|---|---|
| price | 2715 | 471 | 52.4 | 47.9 | 90.2 |
| delivery | 1322 | 287 | 45.3 | 41.3 | 89.5 |
| range | 0 | 237 | 37.0 | 28.8 | 81.0 |
| packaging | 702 | 220 | 28.4 | 25.1 | 88.6 |
| seller | 1143 | 201 | 46.2 | 42.7 | 93.0 |
| speed | 496 | 144 | 23.5 | 17.6 | 76.4 |
| authenticity | 1913 | 142 | 25.7 | 24.7 | 94.4 |
| print | 0 | 138 | 24.7 | 21.2 | 89.9 |
| stability | 0 | 137 | 28.0 | 21.3 | 77.4 |
| quality | 3201 | 111 | 93.8 | 85.8 | 91.0 |
| durability and quality | 0 | 92 | 33.7 | 31.5 | 94.6 |
| firmware | 0 | 62 | 22.1 | 16.6 | 83.9 |
| video quality | 0 | 60 | 26.3 | 22.7 | 86.7 |
| warranty | 110 | 46 | 14.9 | 11.6 | 76.1 |
| setup | 8 | 45 | 13.8 | 13.8 | 88.9 |
