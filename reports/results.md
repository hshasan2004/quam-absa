# Results (test set, best validation epoch)

## EComReviews-BD

| run | split | layout | readout | presence | pair P | pair R | **pair F1** | det F1 | det macro-F1 | sent acc | overall F1 | epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XLM-R (paper) | official | – | – | – | 27.3 | 80.9 | **40.8** | – | – | – | – | – |
| mBERT (paper) | official | – | – | – | 29.8 | 74.7 | **42.6** | – | – | – | – | – |
| BanglaBERT (paper) | official | – | – | – | 31.8 | 77.1 | **45.0** | – | – | – | – | – |
| abl_depth3 | grouped | mmm | memory | mag+value | 38.2 | 38.5 | **38.4** | 42.5 | 10.8 | 86.2 | 68.5 | 14 |
| abl_meanpool | grouped | mm | meanpool | value | 41.4 | 47.3 | **44.2** | 49.7 | 18.1 | 85.8 | 67.2 | 14 |
| abl_meanpool_s1 | grouped | mm | meanpool | value | 43.7 | 49.8 | **46.6** | 52.6 | 17.5 | 85.9 | 67.5 | 15 |
| abl_meanpool_s2 | grouped | mm | meanpool | value | 43.5 | 50.0 | **46.6** | 52.8 | 19.8 | 85.5 | 66.1 | 14 |
| abl_nodomain | grouped | mm | memory | mag+value, no-domain | 36.6 | 46.0 | **40.8** | 45.1 | 13.5 | 85.8 | 67.3 | 15 |
| abl_pres_mag | grouped | mm | memory | mag | 22.1 | 42.9 | **29.2** | 33.8 | 19.8 | 86.2 | 68.6 | 14 |
| abl_pres_val | grouped | mm | memory | value | 43.4 | 50.3 | **46.6** | 51.3 | 17.7 | 86.6 | 68.8 | 14 |
| abl_pres_val_s1 | grouped | mm | memory | value | 49.8 | 46.1 | **47.9** | 53.0 | 14.1 | 86.3 | 69.5 | 14 |
| abl_pres_val_s2 | grouped | mm | memory | value | 39.2 | 44.2 | **41.5** | 46.2 | 15.4 | 86.9 | 69.7 | 15 |
| eco_grouped_cons | grouped | mm | memory | mag+value, cons | 35.8 | 43.1 | **39.1** | 43.3 | 13.0 | 86.8 | 68.9 | 15 |
| eco_grouped_full | grouped | mmsm | memory | mag+value | 28.7 | 31.5 | **30.0** | 33.2 | 10.0 | 83.3 | 56.0 | 1 |
| eco_grouped_mm | grouped | mm | memory | mag+value | 45.9 | 44.8 | **45.4** | 50.1 | 16.7 | 86.1 | 69.2 | 15 |
| eco_grouped_mm_s1 | grouped | mm | memory | mag+value | 53.0 | 46.2 | **49.3** | 53.8 | 14.2 | 86.7 | 70.2 | 15 |
| eco_grouped_mm_s2 | grouped | mm | memory | mag+value | 27.3 | 30.4 | **28.8** | 33.0 | 9.1 | 84.2 | 64.8 | 5 |
| eco_official_full | official | mmsm | memory | mag+value | 37.1 | 42.8 | **39.7** | 44.0 | 14.9 | 86.8 | 69.4 | 15 |
| eco_official_mm | official | mm | memory | mag+value | 51.9 | 52.5 | **52.2** | 57.9 | 20.8 | 86.5 | 74.0 | 14 |
| eco_zeroshot_mm | zeroshot | mm | memory | mag+value | 20.1 | 52.6 | **29.0** | 33.2 | 24.7 | 86.0 | 59.5 | 13 |

## Seed averages (mean ± std over seeds)

| run | seeds | pair F1 | det F1 | det macro-F1 | sent acc | overall F1 |
|---|---|---|---|---|---|---|
| abl_meanpool | 3 | 45.8 ± 1.4 | 51.7 ± 1.7 | 18.5 ± 1.2 | 85.7 ± 0.2 | 66.9 ± 0.7 |
| abl_pres_val | 3 | 45.3 ± 3.3 | 50.2 ± 3.5 | 15.7 ± 1.8 | 86.6 ± 0.3 | 69.3 ± 0.5 |
| eco_grouped_mm | 3 | 41.2 ± 10.9 | 45.7 ± 11.1 | 13.3 ± 3.9 | 85.7 ± 1.3 | 68.1 ± 2.9 |

## BABSA (grouped split)

| run | sent acc | **sent macro-F1** | verbatim F1 | non-verbatim F1 | presence AUROC |
|---|---|---|---|---|---|
| babsa_mm | 63.9 | **64.0** | 64.6 | 60.6 | 79.1 |
