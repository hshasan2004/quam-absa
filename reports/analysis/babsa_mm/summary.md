# Analysis: babsa_mm

split `grouped`, layout `mm`, readout `memory`, presence `mag+value`, seed 42, best epoch 10

## Efficiency (test set, forward pass only, bf16, RTX 2050)

| params | reviews/s | ms/review | aspect queries | peak GPU | memory per review |
|---|---|---|---|---|---|
| 5.78M | 699 | 1.43 | 8,912 | 0.49 GB | 132,096 floats (2 mLSTM blocks, last one queried) |

## BABSA

| sent acc | sent macro-F1 | verbatim F1 | implicit F1 | presence AUROC |
|---|---|---|---|---|
| 63.9 | 64.0 | 64.6 | 60.6 | 79.1 |

Figures: cm_babsa_sentiment, cm_babsa_verbatim, cm_babsa_nonverbatim, roc_babsa_presence

### Confusion (rows gold, columns predicted)

| gold \ pred | positive | neutral | negative |
|---|---|---|---|
| positive | 590 | 198 | 61 |
| neutral | 187 | 493 | 195 |
| negative | 74 | 214 | 558 |

### Per class

| class | P | R | F1 | support |
|---|---|---|---|---|
| positive | 69.3 | 69.5 | 69.4 | 849 |
| neutral | 54.5 | 56.3 | 55.4 | 875 |
| negative | 68.6 | 66.0 | 67.2 | 846 |
| macro avg | 64.1 | 63.9 | 64.0 | 2570 |
| weighted avg | 64.0 | 63.9 | 63.9 | 2570 |

### Verbatim vs implicit aspect terms

| aspects | n | acc | macro-F1 |
|---|---|---|---|
| verbatim | 2060 | 64.4 | 64.6 |
| not verbatim (implicit) | 510 | 61.8 | 60.6 |

### By domain

| domain | n | acc | macro-F1 |
|---|---|---|---|
| Book Review | 780 | 67.7 | 62.4 |
| Politics / Social Issue | 552 | 64.9 | 51.7 |
| General Comment | 318 | 62.6 | 59.4 |
| Personal Opinion | 164 | 59.8 | 58.9 |
| Religious / Spiritual Content | 164 | 62.2 | 55.9 |
| Culture / History | 146 | 54.8 | 50.7 |
| Food Review | 83 | 55.4 | 54.0 |
| Education | 68 | 66.2 | 63.5 |
| Finance / Business | 68 | 64.7 | 61.7 |
| Travel | 51 | 49.0 | 46.2 |
| Entertainment | 43 | 69.8 | 68.3 |
| Sports | 39 | 48.7 | 35.3 |
| Health / Fitness | 28 | 64.3 | 57.8 |
| Customer Service | 23 | 78.3 | 73.0 |
| Technology | 23 | 69.6 | 62.8 |
| Product Review | 9 | 66.7 | 40.0 |
| Music Review | 6 | 66.7 | 55.6 |

### By script

| script | n | acc | macro-F1 |
|---|---|---|---|
| bangla | 2530 | 63.8 | 64.0 |
| mixed | 21 | 90.5 | 83.0 |
| banglish | 16 | 43.8 | 20.3 |
