# Analysis: hyb_babsa

split `grouped`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 10

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 7.23M | 1386 | 0.72 | 8,912 | 0.50 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## BABSA

| sent acc | sent macro-F1 | verbatim F1 | implicit F1 | presence AUROC |
|---|---|---|---|---|
| 64.7 | 64.9 | 65.9 | 59.1 | 84.6 |

Figures: cm_babsa_sentiment, cm_babsa_verbatim, cm_babsa_nonverbatim, roc_babsa_presence

### Confusion (rows gold, columns predicted)

| gold \ pred | positive | neutral | negative |
|---|---|---|---|
| positive | 566 | 219 | 64 |
| neutral | 163 | 496 | 216 |
| negative | 52 | 194 | 600 |

### Per class

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 72.5 | 66.7 | 69.4 | 849 |
| neutral | 54.6 | 56.7 | 55.6 | 875 |
| negative | 68.2 | 70.9 | 69.5 | 846 |
| macro avg | 65.1 | 64.8 | 64.9 | 2570 |
| weighted avg | 65.0 | 64.7 | 64.8 | 2570 |

### Verbatim vs implicit aspect terms

| aspects | n | acc | macro-F1 |
|---|---|---|---|
| verbatim | 2060 | 65.5 | 65.9 |
| not verbatim (implicit) | 510 | 61.2 | 59.1 |

### By domain

| domain | n | acc | macro-F1 |
|---|---|---|---|
| Book Review | 780 | 68.8 | 64.6 |
| Politics / Social Issue | 552 | 65.6 | 49.0 |
| General Comment | 318 | 64.8 | 62.2 |
| Personal Opinion | 164 | 55.5 | 55.3 |
| Religious / Spiritual Content | 164 | 65.9 | 61.9 |
| Culture / History | 146 | 54.8 | 51.7 |
| Food Review | 83 | 45.8 | 45.2 |
| Education | 68 | 67.6 | 66.0 |
| Finance / Business | 68 | 66.2 | 60.3 |
| Travel | 51 | 52.9 | 50.7 |
| Entertainment | 43 | 72.1 | 71.2 |
| Sports | 39 | 71.8 | 61.1 |
| Health / Fitness | 28 | 50.0 | 50.0 |
| Customer Service | 23 | 82.6 | 77.8 |
| Technology | 23 | 56.5 | 47.1 |
| Product Review | 9 | 88.9 | 86.2 |
| Music Review | 6 | 83.3 | 61.9 |

### By script

| script | n | acc | macro-F1 |
|---|---|---|---|
| bangla | 2530 | 64.4 | 64.7 |
| mixed | 21 | 90.5 | 77.5 |
| banglish | 16 | 68.8 | 55.2 |
