# Karsilastirma Ozeti

- CNN2+CBAM : `runs/cnn2_cbam_20260511-103338_cbam_v1`
- CNN2+CBAM (tuned) : `runs/cnn2_cbam_20260509-182442_tuned`

## Ana Metrikler

| Metrik | CNN2+CBAM | CNN2+CBAM (tuned) | Δ (pp) |
|---|---:|---:|---:|
| Overall accuracy | 0.5160 | 0.5038 | -1.22 |
| Avg acc (-20..-10 dB) | 0.1135 | 0.1127 | -0.08 |
| Avg acc (-10..0 dB)   | 0.4652 | 0.4525 | -1.27 |
| Avg acc (0..18 dB)    | 0.7739 | 0.7551 | -1.88 |

## QAM16 ↔ QAM64 Confusion (SNR >= 0 dB)

| Olay | CNN2+CBAM | CNN2+CBAM (tuned) |
|---|---:|---:|
| QAM16 dogru tahmin | 0.6715 | 0.7045 |
| QAM64 dogru tahmin | 0.3465 | 0.2815 |
| QAM16 → QAM64 hata | 0.2555 | 0.1950 |
| QAM64 → QAM16 hata | 0.6040 | 0.6495 |
