# Karsilastirma Ozeti

- CNN2+CBAM (default) : `runs/cnn2_cbam_20260511-103338_cbam_v1`
- CNN2+CBAM (Optuna tuned) : `/content/drive/MyDrive/amcprojects/amc_project/runs/cnn2_cbam_20260511-104549_top3_trial11`

## Ana Metrikler

| Metrik | CNN2+CBAM (default) | CNN2+CBAM (Optuna tuned) | Δ (pp) |
|---|---:|---:|---:|
| Overall accuracy | 0.5160 | 0.5244 | +0.84 |
| Avg acc (-20..-10 dB) | 0.1135 | 0.1106 | -0.29 |
| Avg acc (-10..0 dB)   | 0.4652 | 0.4545 | -1.06 |
| Avg acc (0..18 dB)    | 0.7739 | 0.7996 | +2.57 |

## QAM16 ↔ QAM64 Confusion (SNR >= 0 dB)

| Olay | CNN2+CBAM (default) | CNN2+CBAM (Optuna tuned) |
|---|---:|---:|
| QAM16 dogru tahmin | 0.6715 | 0.7655 |
| QAM64 dogru tahmin | 0.3465 | 0.1215 |
| QAM16 → QAM64 hata | 0.2555 | 0.0900 |
| QAM64 → QAM16 hata | 0.6040 | 0.7765 |
