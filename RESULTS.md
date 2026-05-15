# AMC Project — Sonuçlar Özeti

**Proje:** Automatic Modulation Classification on RadioML 2016.10A
**Grup:** Ulaş Gürses, İlhan Deger, Semih Öncel, Alptürk İncesu
**Sunum Tarihi:** 12 Mayıs 2026

---

## 1. Genel Bakış

Bu projede DeepSig RadioML 2016.10A veri seti (220.000 örnek, 11 modülasyon, -20…+18 dB SNR) üzerinde **dört aşamalı bir geliştirme yolculuğu** gerçekleştirdik:

1. **Baseline reproduksiyon** (CNN2 — AMR-Benchmark stili)
2. **Architectural improvement** (CNN2 + CBAM attention)
3. **Hyperparameter optimization** (Optuna Bayesian search + top-3 rerun)
4. **Top-3 model ensemble**

Toplam iyileştirme: **+3.46 puan** baseline'a göre.

---

## 2. Veri Hazırlığı

| Aşama | İşlem | Sonuç |
|---|---|---|
| Pickle yükleme | Python 2 pickle (`encoding='latin1'`) | 220.000 × (2, 128) örnek |
| Normalizasyon | Per-sample L2 norm | Skala invarianti |
| Bölme | **(mod × SNR) joint stratified** 60/20/20 | Train: 132k / Val: 44k / Test: 44k |
| Reprodüksiyon | Seed = 42, `cudnn.deterministic=True` | Tekrarlanabilir |

**Önemli:** Standart pratik sadece SNR'a göre stratifikasyon önerir. Biz daha güçlü bir koruma uyguladık: 11 mod × 20 SNR = 220 bucket'a göre stratification. Her split tüm bucket'ları eşit oranda içerir.

---

## 3. Modeller

### 3.1 CNN2 (Baseline)
AMR-Benchmark çalışmasındaki referans model.
- **Mimari:** Conv1d(2→256) → ReLU → Dropout → Conv1d(256→80) → ReLU → Dropout → FC(10240→256) → FC(256→11)
- **Parametreler:** ~2.69M
- **Default hyperparams:** lr=1e-3, dropout=0.5, weight_decay=1e-4, batch=256, AdamW

### 3.2 CNN2 + CBAM
Her conv katmanından sonra CBAM (Convolutional Block Attention Module) eklendi.
- **Channel Attention:** AvgPool + MaxPool → MLP(C → C/r → C) → sigmoid (r=8 default)
- **Spatial Attention:** [avg, max] → Conv1d(2→1, k=7) → sigmoid
- **Parametreler:** ~2.70M (sadece ~0.4% artış)

### 3.3 CNN2 + CBAM (Optuna Tuned)
15-trial Bayesian optimization + top-3 rerun ile bulunan en iyi config:
- lr = **0.001641**
- dropout = **0.6**
- weight_decay = **1e-5**
- CBAM reduction = **16**

### 3.4 Top-3 Ensemble
Optuna top-3 trial'larının 30 epoch'lık tam eğitimlerinden softmax probability averaging.

---

## 4. Eğitim Setup'ı

| Parametre | Değer |
|---|---|
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Mixed Precision (AMP) | ✓ Açık |
| Batch size | 256 |
| Epochs | 30 |
| Early stopping | patience=10 (val_loss) |
| Hardware | NVIDIA Tesla T4 (Google Colab) |

**AMP etkisi:** Eğitim süresi yarı yarıya düştü. CNN2: ~3 dk, CBAM: ~4 dk (30 epoch).

---

## 5. Hyperparameter Optimization Detayları

### 5.1 Search space
| Parametre | Aralık | Tip |
|---|---|---|
| `lr` | 1e-4 — 5e-3 | log-uniform |
| `dropout` | {0.3, 0.4, 0.5, 0.6} | categorical |
| `weight_decay` | {1e-5, 1e-4, 1e-3} | categorical |
| `reduction` (CBAM) | {4, 8, 16} | categorical |

### 5.2 Optuna ayarları
- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: MedianPruner (warmup=2 epoch, startup=3 trial)
- Objective: SNR ≥ 0 dB validation accuracy
- Trial başına: 8 epoch + AMP
- Persistent storage: SQLite (kopuş-dirençli)

### 5.3 Sonuçlar
- **15 trial:** 8 tamamlandı, 7 prune edildi
- **Hyperparameter importance:**
  - lr: **0.71** (dominant)
  - dropout: 0.13
  - weight_decay: 0.08
  - reduction: 0.08

### 5.4 Proxy Bias bulgusu
8-epoch'lık short-training ile elde edilen "en iyi" konfigürasyon, 30-epoch'lık full training'de **default'tan kötü** çıktı (0.5038 vs 0.5160). Bu, klasik **proxy objective bias** olgusunun canlı bir örneği. Bu nedenle top-3 trial'ı 30 epoch tam eğitimde "yarıştırdık":
- Top-1 (trial #11): **0.5244** ← yeni en iyi
- Top-2 (trial #8): 0.5045
- Top-3 (trial #14): 0.5038 (= Optuna'nın final eğitiminin sonucu)

Top-1, default CBAM'i +0.84 puan geçti.

---

## 6. Final Sonuçlar — Ana Tablo

### 6.1 Test accuracy karşılaştırması

| Model | Test Acc | Δ vs Baseline |
|---|---:|---:|
| CNN2 baseline | 0.4999 | — |
| CNN2 + CBAM (default) | 0.5160 | +1.61 pp |
| CNN2 + CBAM (Optuna 8-epoch best) | 0.5038 | +0.39 pp |
| CNN2 + CBAM (Top-1 tuned) | 0.5244 | +2.45 pp |
| **Top-3 Ensemble** | **0.5345** | **+3.46 pp** ⭐ |

### 6.2 SNR bölgesi bazlı analiz

| SNR aralığı | Baseline | Default CBAM | Tuned CBAM | Ensemble |
|---|---:|---:|---:|---:|
| Çok düşük (-20 ile -10 dB) | 0.108 | 0.114 | ~0.11 | 0.114 |
| Düşük (-10 ile 0 dB) | 0.455 | 0.465 | ~0.46 | 0.470 |
| **Yüksek (0 ile +18 dB)** | **0.750** | **0.774** | **0.804** | **0.821** |

🎯 **En büyük kazanım yüksek SNR plateau'sunda:** Ensemble %82.1 ile baseline'ı **+7.1 puan** geçiyor.

### 6.3 Per-SNR karşılaştırma (Ensemble vs Baseline)

| SNR | Baseline | Ensemble | Δ |
|---:|---:|---:|---:|
| -20 | 0.090 | 0.093 | +0.3 |
| -10 | 0.155 | 0.161 | +0.6 |
| -6 | 0.391 | 0.406 | +1.5 |
| -2 | 0.669 | 0.677 | +0.8 |
| 0 | 0.726 | 0.749 | **+2.3** |
| +6 | 0.755 | 0.826 | **+7.1** |
| +12 | 0.741 | 0.821 | **+8.0** |
| +18 | 0.756 | 0.828 | **+7.2** |

**Pattern:** Düşük SNR'da gürültü dominant, ensemble da çare değil. Yüksek SNR'da ensemble dramatik fark yaratıyor — modülasyonlar arası ince yapısal farkları daha iyi ayırt ediyor.

---

## 7. QAM-16 ↔ QAM-64 Confusion Analizi

Proposal'ın anahtar iddiası: **"high misclassification rates between overlapping high-order modulations like 16QAM and 64QAM"** — buna karşı çözüm olarak attention mekanizması.

### 7.1 Baseline davranışı (problematik)
SNR ≥ 0 dB için:
- QAM16 doğruluk: **%84** (yüksek)
- QAM64 doğruluk: **%10** 🚨
- QAM64 → QAM16 yanlış sınıflandırma: **%85** 🚨

**Yorum:** Baseline neredeyse her şüpheli QAM örneğine "QAM16" diyor. Tek yönlü bias.

### 7.2 Default CBAM'le iyileşme
- QAM16 doğruluk: %84 → %67 (hafif düşüş)
- QAM64 doğruluk: %10 → **%35** (**3.5x iyileşme**)
- QAM64 → QAM16 hata: %85 → %60 (**-25 puan**)

CBAM, baseline'ın bias'ını düzeltti. İki sınıf arasındaki **karar sınırını dengeledi**.

### 7.3 Tüm modellerin tam tablosu

SNR ≥ 0 dB için QAM confusion oranları:

| Olay | Baseline | Default CBAM | Tuned CBAM | Top-3 Ensemble |
|---|---:|---:|---:|---:|
| QAM16 → QAM64 hata | 0.08 | 0.26 | 0.09 | 0.07 |
| QAM64 → QAM16 hata | **0.85** 🚨 | 0.60 | 0.78 | 0.76 |
| QAM16 doğru | 0.84 | 0.67 | 0.77 | 0.82 |
| QAM64 doğru | **0.10** 🚨 | **0.35** ✓ | 0.12 | 0.17 |

### 7.4 Şaşırtıcı bulgu: Optimization ve QAM dengesi trade-off'u

**Default CBAM'in en güçlü yanı QAM dengesini düzeltmesiydi** (QAM64: 10% → 35%).

Ancak Optuna ve Ensemble bu dengeyi kısmen geri kaybettiler:
- Tuned QAM64 doğruluk: %12 (baseline'a yakın)
- Ensemble QAM64 doğruluk: %17 (baseline'dan iyi ama default CBAM'den düşük)

**Olası nedenler:**

1. **Objective bias:** Optuna `SNR ≥ 0 dB overall accuracy`'i optimize etti — QAM dengesi bu metriğin içinde "gizli kaldı". Sınıf-bazlı F1 veya balanced accuracy kullansaydık farklı sonuç alabilirdik.

2. **Aggressive regularization:** Tuned config dropout=0.6 (default 0.5). Daha güçlü regularization, modelin **az nüfuslu** sınıflarda (QAM64'ün yüksek SNR'da net karar gerektiren yapısı) daha çekingen karar vermesine yol açmış olabilir.

3. **Trade-off gerçek:** Overall accuracy'i %1.85 puan artırırken (default→ensemble), bu kazanımın bir kısmı kolay sınıflardan (QAM16, AM-DSB) gelmiş olabilir. Zor sınıflarda (QAM64) ise ya değişim yok ya da hafif kötüleşme.

### 7.5 Sunum/Rapor için doğru framing

**Üç model, üç farklı use-case:**

| Use case | Önerilen model | Test acc | QAM64 acc |
|---|---|---:|---:|
| Maksimum overall accuracy | **Top-3 Ensemble** | 0.5345 | 0.17 |
| Dengeli QAM ayrımı | **Default CBAM** | 0.5160 | 0.35 |
| Reproduksiyon baseline | CNN2 | 0.4999 | 0.10 |

**Anahtar mesaj:** "Mimari iyileştirmeler (CBAM) modülasyon dengesini düzeltir. Optimization sonrası iyileştirmeler overall accuracy'yi artırır ama bu, kolay sınıfların ağırlığının artışıyla gerçekleşebilir. Bu, ML projelerinde **objective seçiminin kritikliğini** gösteren bir bulgudur."

### 7.6 Future work: Multi-objective optimization
Bu deneyim gösterdi ki, **overall accuracy + QAM-balanced metric** ikilisini birlikte optimize eden multi-objective Optuna study (NSGA-II sampler) Pareto front'unda farklı modeller bulur. Bunu future work olarak öneriyoruz.

---

## 8. Modülasyon-bazlı Detaylı Analiz

CBAM'in iyileştirdiği iki büyük dengesizlik:

### 8.1 QAM ailesi (yukarıda detaylı)
QAM16/QAM64 dengesizliği düzeltildi.

### 8.2 Analog modülasyonlar (AM-DSB ↔ WBFM)
- Baseline'da AM-DSB'nin %41'i WBFM olarak yanlış sınıflandırılıyordu
- Default CBAM'de bu oran **%15'e** düştü (-26 puan!)
- AM-DSB doğruluk: %59 → %85 (+26 puan)

### 8.3 Zaten mükemmel olan sınıflar (her iki modelde de)
- AM-SSB: ~%97-99
- BPSK: ~%98-99
- CPFSK: ~%100
- GFSK: ~%96
- PAM4: ~%96-98

### 8.4 Hâlâ zorlu olanlar
- 8PSK ↔ QPSK karışıklığı (her iki sınıf da diğerine bakıyor)
- Düşük SNR'larda tüm sınıflar (zaten beklenen)

---

## 9. Üretilen Tüm Çıktılar

### 9.1 Eğitim çıktıları (`runs/`)
- `cnn2_*_baseline/best.pt` — CNN2 baseline checkpoint
- `cnn2_cbam_*_cbam_v1/best.pt` — Default CBAM checkpoint
- `cnn2_cbam_*_top1_trial11/best.pt` — Optuna top-1 (final tuned model)
- `cnn2_cbam_*_top2_trial8/best.pt` — Optuna top-2
- `cnn2_cbam_*_top3_trial14/best.pt` — Optuna top-3
- `cnn2_cbam_*_tuned/best.pt` — Optuna 8-epoch final (proxy bias örneği)

### 9.2 Bireysel evaluate çıktıları (`figures/<run_name>/`)
Her run için:
- `acc_vs_snr.png`
- `confusion_overall.png`
- `confusion_highSNR.png`
- `cm_overall.npy`, `cm_highSNR.npy`
- `metrics.json`

### 9.3 Karşılaştırma figürleri (`figures/comparison_*/`)
Üç adet:
- **`comparison_ensemble_vs_baseline/`** — sunumun ANA figürü (+3.46 pp)
- `comparison_tuned_vs_default/` — Optuna kazancı (+0.84 pp)
- `comparison_ensemble_vs_tuned/` — ensemble kazancı (+1.01 pp)

Her klasörde:
- `acc_vs_snr_overlay.png`
- `confusion_side_by_side.png`
- `qam_confusion_bar.png`
- `summary.json`, `summary.md`

### 9.4 Optuna görselleştirmeleri (`figures/`)
- `optuna_history.png` — trial-by-trial optimization
- `optuna_importances.png` — lr dominant (%71)
- `optuna_parallel.png` — parallel coordinate plot

### 9.5 Ensemble (`figures/ensemble_top3/`)
- `acc_vs_snr.png`, `confusion_overall.png`, `confusion_highSNR.png`
- `metrics.json`, `cm_*.npy`

### 9.6 EDA (`figures/eda/`)
- `class_balance.png` — split balance kontrolü
- `constellations_snr18.png`, `constellations_snr0.png` — IQ scatter
- `snr_progression_QAM16.png`, `snr_progression_QAM64.png`, `snr_progression_QPSK.png` — SNR'a göre constellation evrimi
- `iq_timeseries_snr18.png` — zaman serisi I/Q

---

## 10. Sunum/Rapor İçin Anahtar Mesajlar

### Mesaj 1: Reproduce ve Extend uyguladık
"AMR-Benchmark çalışmasını başlangıç noktası olarak aldık, CNN2 baseline'ı %49.99 ile başarıyla reproduce ettik."

### Mesaj 2: CBAM mimari iyileştirme sağlıyor — özellikle QAM dengesi
"Channel + Spatial attention'ın 1D versiyonunu IQ sinyaline uyguladık. Default CBAM +1.61 puan kazandırdı, ama asıl önemli olan **modülasyon-içi yapısal benzerlikleri çözmesi** — özellikle QAM64 doğruluğunu %10'dan %35'e çıkardı (3.5x), AM-DSB doğruluğunu %59'dan %85'e taşıdı."

### Mesaj 2b: ⚠️ Optimization trade-off — bilimsel olarak değerli
"Önemli bir bulgu: Optuna optimization overall accuracy'yi artırırken, default CBAM'in QAM-dengeleyici davranışını kısmen feda etti (QAM64 acc: 35% → 17%). Bu, single-objective optimization'ın multi-faceted ML problemlerindeki **gizli trade-off'larını** ortaya koyuyor. Future work: multi-objective optimization."

### Mesaj 3: Hyperparameter optimization scientific rigor sağlıyor
"15-trial Bayesian optimization (Optuna) gerçekleştirdik. Importance analysis lr'ın %71 dominant olduğunu gösterdi. Önemli bir bilimsel bulgu: 8-epoch proxy objective ile elde edilen 'en iyi' config 30-epoch full training'de **default'tan kötü** çıktı — bu, klasik **proxy bias** olgusunun bir örneği. Top-3 rerun ile bias'ı bypass ettik."

### Mesaj 4: Ensemble extra performance verir
"Top-3 modelin softmax probability ortalaması (model ensemble), bireysel en iyi modeli +1.01 puan geçti. Yüksek SNR plateau'sunda accuracy %77'lerden **%82'lere** çıktı (+5 puan)."

### Mesaj 5: Toplam etki
"Baseline → Final Model: **+3.46 puan** test accuracy iyileştirmesi. Yüksek SNR'da +7 puan'a varan kazanımlar. Proposal vaadini fazlasıyla aştık."

---

## 11. Bilimsel Değer (Sunum Sonu Slaydı için)

### Pozitif bulgular
1. CBAM, modülasyon-içi yapısal benzerlikleri çözmede etkili (QAM, analog modülasyonlar).
2. Yüksek SNR plateau'sundaki iyileşme (+7 pp) büyük teorik öneme sahip — gürültü değil, modülasyon ailesi karışıklığı asıl darboğazmış.
3. Optuna importance analysis pratik öneri veriyor: lr'a odaklan.

### Negative result (bilimsel olarak değerli)
1. **Proxy bias:** Kısa eğitim'in en iyisi, uzun eğitim'in en iyisi olmayabilir. Her ML projesi için önemli ders.
2. **Çok düşük SNR'da (-20 ile -10 dB) attention da çare değil** — gürültü gerçekten dominant. Bu, modülasyon klasifikasyonunun teorik sınırlarına işaret ediyor.
3. **Single-objective optimization trade-off'u:** Overall accuracy'i optimize ederken sınıf dengesi feda edilebilir. Default CBAM QAM dengesini düzeltmişti, Optuna+Ensemble bunu kısmen geri kaybetti. ML pipeline tasarımında **objective seçiminin** önemini gösterir.

### Future work
1. Daha geniş search space + uzun trial epochs (proxy bias'tan tamamen kaçınmak için)
2. **Multi-objective optimization** (NSGA-II sampler ile) — overall accuracy + balanced/macro F1 birlikte optimize, Pareto front analizi
3. Data augmentation (phase rotation, frequency offset injection)
4. ResNet1D + CBAM gibi daha derin mimariler
5. Ensemble'a farklı mimari türleri ekleme (mimari çeşitlilik)
6. Class-weighted loss fonksiyonu — QAM gibi zor sınıflar için ekstra ağırlık

---

## 12. Çalıştırma Talimatları (Reprodüksiyon)

Sıralı:
```bash
python data_loader.py                                              # bir kere
python train.py --model cnn2 --epochs 30 --tag baseline
python train.py --model cnn2_cbam --epochs 30 --tag cbam_v1
python tune.py --n_trials 15 --epochs_per_trial 8 --final_epochs 30
python rerun_top3.py --epochs 30
python ensemble.py
python evaluate.py --ckpt runs/cnn2_<ts>_baseline/best.pt
python evaluate.py --ckpt runs/cnn2_cbam_<ts>_cbam_v1/best.pt
python evaluate.py --ckpt runs/cnn2_cbam_<ts>_top1_trial11/best.pt
python compare.py --baseline runs/cnn2_<ts>_baseline --cbam_metrics figures/ensemble_top3 \
                  --baseline_label "CNN2 (baseline)" --cbam_label "Top-3 Ensemble" \
                  --out_subdir comparison_ensemble_vs_baseline
```

Toplam GPU süresi (Tesla T4, AMP açık): **~2 saat**.

---

*Son güncelleme: 10 Mayıs 2026*
