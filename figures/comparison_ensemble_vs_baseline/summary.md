# Karsilastirma Ozeti

- CNN2 (baseline) : `runs/cnn2_20260511-103116_baseline`
- Top-3 Ensemble : `figures/ensemble_top3`

## Ana Metrikler

| Metrik | CNN2 (baseline) | Top-3 Ensemble | Δ (pp) |
|---|---:|---:|---:|
| Overall accuracy | 0.4999 | 0.5345 | +3.46 |
| Avg acc (-20..-10 dB) | 0.1079 | 0.1123 | +0.44 |
| Avg acc (-10..0 dB)   | 0.4547 | 0.4653 | +1.06 |
| Avg acc (0..18 dB)    | 0.7503 | 0.8133 | +6.30 |

## QAM16 ↔ QAM64 Confusion (SNR >= 0 dB)

| Olay | CNN2 (baseline) | Top-3 Ensemble |
|---|---:|---:|
| QAM16 dogru tahmin | 0.8445 | 0.8235 |
| QAM64 dogru tahmin | 0.0960 | 0.1660 |
| QAM16 → QAM64 hata | 0.0835 | 0.0730 |
| QAM64 → QAM16 hata | 0.8545 | 0.7585 |
