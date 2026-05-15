# Karsilastirma Ozeti

- CNN2+CBAM (tuned) : `/content/drive/MyDrive/amcprojects/amc_project/runs/cnn2_cbam_20260511-104549_top3_trial11`
- Top-3 Ensemble : `figures/ensemble_top3`

## Ana Metrikler

| Metrik | CNN2+CBAM (tuned) | Top-3 Ensemble | Δ (pp) |
|---|---:|---:|---:|
| Overall accuracy | 0.5244 | 0.5345 | +1.01 |
| Avg acc (-20..-10 dB) | 0.1106 | 0.1123 | +0.17 |
| Avg acc (-10..0 dB)   | 0.4545 | 0.4653 | +1.08 |
| Avg acc (0..18 dB)    | 0.7996 | 0.8133 | +1.37 |

## QAM16 ↔ QAM64 Confusion (SNR >= 0 dB)

| Olay | CNN2+CBAM (tuned) | Top-3 Ensemble |
|---|---:|---:|
| QAM16 dogru tahmin | 0.7655 | 0.8235 |
| QAM64 dogru tahmin | 0.1215 | 0.1660 |
| QAM16 → QAM64 hata | 0.0900 | 0.0730 |
| QAM64 → QAM16 hata | 0.7765 | 0.7585 |
