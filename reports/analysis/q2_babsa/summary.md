# Analysis: q2_babsa

split `grouped`, layout `mm`, readout `hybrid`, presence `mag+value`, seed 42, best epoch 7

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 10.04M | 1154 | 0.87 | 8,912 | 0.52 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## BABSA

| sent acc | sent macro-F1 | verbatim F1 | implicit F1 | presence AUROC |
|---|---|---|---|---|
| 63.4 | 63.4 | 64.0 | 59.8 | 89.0 |

Figures: cm_babsa_sentiment, cm_babsa_verbatim, cm_babsa_nonverbatim, roc_babsa_presence

### Confusion (rows gold, columns predicted)

| gold \ pred | positive | neutral | negative |
|---|---|---|---|
| positive | 612 | 182 | 55 |
| neutral | 216 | 477 | 182 |
| negative | 96 | 210 | 540 |

### Per class

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 66.2 | 72.1 | 69.0 | 849 |
| neutral | 54.9 | 54.5 | 54.7 | 875 |
| negative | 69.5 | 63.8 | 66.5 | 846 |
| macro avg | 63.5 | 63.5 | 63.4 | 2570 |
| weighted avg | 63.4 | 63.4 | 63.3 | 2570 |

### Verbatim vs implicit aspect terms

| aspects | n | acc | macro-F1 |
|---|---|---|---|
| verbatim | 2060 | 63.7 | 64.0 |
| not verbatim (implicit) | 510 | 62.0 | 59.8 |

### By domain

| domain | n | acc | macro-F1 |
|---|---|---|---|
| Book Review | 780 | 67.7 | 62.1 |
| Politics / Social Issue | 552 | 61.2 | 51.0 |
| General Comment | 318 | 62.9 | 61.2 |
| Personal Opinion | 164 | 56.1 | 55.0 |
| Religious / Spiritual Content | 164 | 65.9 | 59.8 |
| Culture / History | 146 | 54.8 | 49.5 |
| Food Review | 83 | 49.4 | 47.8 |
| Education | 68 | 63.2 | 60.4 |
| Finance / Business | 68 | 55.9 | 50.4 |
| Travel | 51 | 58.8 | 59.4 |
| Entertainment | 43 | 72.1 | 69.6 |
| Sports | 39 | 69.2 | 54.4 |
| Health / Fitness | 28 | 75.0 | 75.7 |
| Customer Service | 23 | 73.9 | 69.4 |
| Technology | 23 | 87.0 | 83.8 |
| Product Review | 9 | 66.7 | 58.5 |
| Music Review | 6 | 66.7 | 48.9 |

### By script

| script | n | acc | macro-F1 |
|---|---|---|---|
| bangla | 2530 | 63.2 | 63.3 |
| mixed | 21 | 90.5 | 77.8 |
| banglish | 16 | 50.0 | 38.9 |
