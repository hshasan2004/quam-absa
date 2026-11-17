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
| ens_hyb_official | official | mm | hybrid | mag+value | 55.2 | 51.4 | **53.3** | 58.8 | 19.3 | 87.6 | 73.1 | – |
| ens_hyb_zeroshot | zeroshot | mm | hybrid | mag+value | 44.4 | 39.8 | **42.0** | 46.8 | 20.9 | 86.2 | 61.7 | – |
| ens_hybrid | grouped | mm | hybrid | mag+value | 51.3 | 50.7 | **51.0** | 56.1 | 18.2 | 87.8 | 70.0 | – |
| ens_meanpool | grouped | mm | meanpool | value | 45.1 | 50.4 | **47.6** | 53.4 | 18.1 | 86.6 | 68.5 | – |
| ens_memory | grouped | mm | memory | mag+value | 55.0 | 44.7 | **49.3** | 53.8 | 12.0 | 87.8 | 71.2 | – |
| fin_hybrid | grouped | mm | hybrid | mag+value | 40.7 | 50.1 | **44.9** | 50.9 | 19.8 | 86.0 | 68.5 | 15 |
| fin_hybrid_s1 | grouped | mm | hybrid | mag+value | 53.6 | 46.0 | **49.5** | 54.7 | 18.0 | 86.6 | 68.1 | 12 |
| fin_hybrid_s2 | grouped | mm | hybrid | mag+value | 46.4 | 47.9 | **47.1** | 53.0 | 17.7 | 86.2 | 68.7 | 15 |
| fin_meanpool | grouped | mm | meanpool | value | 41.4 | 47.3 | **44.2** | 49.7 | 18.1 | 85.8 | 67.2 | 14 |
| fin_meanpool_s1 | grouped | mm | meanpool | value | 43.7 | 49.8 | **46.6** | 52.6 | 17.5 | 85.9 | 67.5 | 15 |
| fin_meanpool_s2 | grouped | mm | meanpool | value | 43.5 | 50.0 | **46.6** | 52.8 | 19.8 | 85.5 | 66.1 | 14 |
| fin_memory | grouped | mm | memory | mag+value | 45.9 | 44.8 | **45.4** | 50.1 | 16.7 | 86.1 | 69.2 | 15 |
| fin_memory_s1 | grouped | mm | memory | mag+value | 53.0 | 46.2 | **49.3** | 53.8 | 14.2 | 86.7 | 70.2 | 15 |
| fin_memory_s2 | grouped | mm | memory | mag+value | 30.2 | 42.0 | **35.1** | 39.1 | 14.3 | 86.3 | 69.1 | 13 |
| fin_pres_val | grouped | mm | memory | value | 43.4 | 50.3 | **46.6** | 51.3 | 17.7 | 86.6 | 68.8 | 14 |
| fin_pres_val_s1 | grouped | mm | memory | value | 49.8 | 46.1 | **47.9** | 53.0 | 14.1 | 86.3 | 69.5 | 14 |
| fin_pres_val_s2 | grouped | mm | memory | value | 39.2 | 44.2 | **41.5** | 46.2 | 15.4 | 86.9 | 69.7 | 15 |
| fin_tied | grouped | mm | memory | mag+value | 44.6 | 39.7 | **42.0** | 46.1 | 13.0 | 85.9 | 70.0 | 12 |
| fin_tied_s1 | grouped | mm | memory | mag+value | 54.0 | 40.0 | **45.9** | 50.0 | 12.1 | 86.8 | 68.9 | 15 |
| fin_tied_s2 | grouped | mm | memory | mag+value | 28.6 | 49.1 | **36.2** | 40.7 | 20.7 | 86.3 | 69.8 | 15 |
| hyb_official | official | mm | hybrid | mag+value | 41.5 | 50.7 | **45.6** | 51.3 | 19.1 | 86.4 | 70.1 | 14 |
| hyb_official_s1 | official | mm | hybrid | mag+value | 56.0 | 47.5 | **51.4** | 56.8 | 18.5 | 86.9 | 72.8 | 13 |
| hyb_official_s2 | official | mm | hybrid | mag+value | 56.0 | 46.6 | **50.9** | 56.8 | 17.4 | 86.6 | 72.0 | 13 |
| hyb_zeroshot | zeroshot | mm | hybrid | mag+value | 40.3 | 41.1 | **40.7** | 46.4 | 21.1 | 84.7 | 59.9 | 13 |
| hyb_zeroshot_s1 | zeroshot | mm | hybrid | mag+value | 31.0 | 46.4 | **37.2** | 42.9 | 24.9 | 85.1 | 61.6 | 15 |
| hyb_zeroshot_s2 | zeroshot | mm | hybrid | mag+value | 45.9 | 39.9 | **42.7** | 48.6 | 21.6 | 84.8 | 61.2 | 15 |

## Seed averages (mean ± std over seeds)

| run | seeds | pair F1 | det F1 | det macro-F1 | sent acc | overall F1 |
|---|---|---|---|---|---|---|
| abl_meanpool | 3 | 45.8 ± 1.4 | 51.7 ± 1.7 | 18.5 ± 1.2 | 85.7 ± 0.2 | 66.9 ± 0.7 |
| abl_pres_val | 3 | 45.3 ± 3.3 | 50.2 ± 3.5 | 15.7 ± 1.8 | 86.6 ± 0.3 | 69.3 ± 0.5 |
| eco_grouped_mm | 3 | 41.2 ± 10.9 | 45.7 ± 11.1 | 13.3 ± 3.9 | 85.7 ± 1.3 | 68.1 ± 2.9 |
| fin_hybrid | 3 | 47.2 ± 2.3 | 52.9 ± 2.0 | 18.5 ± 1.1 | 86.3 ± 0.3 | 68.4 ± 0.3 |
| fin_meanpool | 3 | 45.8 ± 1.4 | 51.7 ± 1.7 | 18.5 ± 1.2 | 85.7 ± 0.2 | 66.9 ± 0.7 |
| fin_memory | 3 | 43.3 ± 7.3 | 47.7 ± 7.7 | 15.1 ± 1.4 | 86.4 ± 0.3 | 69.5 ± 0.6 |
| fin_pres_val | 3 | 45.3 ± 3.3 | 50.2 ± 3.5 | 15.7 ± 1.8 | 86.6 ± 0.3 | 69.3 ± 0.5 |
| fin_tied | 3 | 41.4 ± 4.9 | 45.6 ± 4.7 | 15.3 ± 4.8 | 86.3 ± 0.5 | 69.6 ± 0.6 |
| hyb_official | 3 | 49.3 ± 3.2 | 54.9 ± 3.2 | 18.3 ± 0.9 | 86.6 ± 0.3 | 71.6 ± 1.4 |
| hyb_zeroshot | 3 | 40.2 ± 2.8 | 45.9 ± 2.9 | 22.6 ± 2.0 | 84.8 ± 0.2 | 60.9 ± 0.9 |

## BABSA (grouped split)

| run | sent acc | **sent macro-F1** | verbatim F1 | non-verbatim F1 | presence AUROC |
|---|---|---|---|---|---|
| babsa_mm | 63.9 | **64.0** | 64.6 | 60.6 | 79.1 |
| ens_hyb_babsa | 65.5 | **65.7** | 66.5 | 60.9 | 86.0 |
| hyb_babsa | 64.7 | **64.9** | 65.9 | 59.1 | 84.6 |
| hyb_babsa_s1 | 62.3 | **62.5** | 63.8 | 56.3 | 85.4 |
| hyb_babsa_s2 | 63.5 | **63.6** | 64.3 | 58.8 | 85.2 |
