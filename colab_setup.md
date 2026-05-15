# Google Colab — Kurulum Rehberi

Bu projeyi Colab'da çalıştırmak için tek seferlik kurulum.

## 1. Drive hazırlığı (yerel bilgisayardan, bir kere)

`amc_project/` klasörünü **olduğu gibi** Google Drive'a yükle. Önerilen yer:

```
MyDrive/
└── amc_project/
    ├── config.py
    ├── data_loader.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    ├── visualize.py
    ├── amc_colab.ipynb
    ├── requirements.txt
    └── data/
        └── RML2016.10a_dict.pkl   <-- 640 MB, manuel olarak buraya koy
```

Yükleme yöntemleri:
- **Web arayüzünden:** drive.google.com → "Yeni" → "Klasör yükle". Klasör boyutu küçük olduğu için (~25 KB kod + 640 MB pickle) sürükle-bırak yeterli.
- **Drive Desktop:** Drive'ı senkronize ediyorsan klasörü doğrudan `My Drive\amc_project\` içine kopyala, otomatik yüklenir.

## 2. Dataset

`RML2016.10a_dict.pkl` (~640 MB) dosyasını DeepSig'in resmi sayfasından indir:

https://www.deepsig.ai/datasets

Kayıt olup "RadioML 2016.10A" datasetini seçince `.tar.bz2` indirir. Açtığında içinde `RML2016.10a_dict.pkl` dosyası vardır. Bunu `MyDrive/amc_project/data/` içine yükle.

## 3. Notebook'u aç

Drive'da `amc_project/amc_colab.ipynb` dosyasına çift tıkla → "Open with Colab".

## 4. Runtime ayarı

Colab menüden:
- **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 yeterli)
- "Save"

## 5. Hücreleri sırayla çalıştır

Notebook'taki bölümler sırayla:
1. Drive'ı bağla, GPU'yu doğrula
2. Bağımlılıkları yükle
3. Veri ön-işleme (~2-3 dk, bir kere)
4. EDA görselleri
5. TensorBoard inline aç
6. Baseline (CNN2) eğit (~15-25 dk, 30 epoch)
7. CBAM modelini eğit (~20-30 dk, 30 epoch)
8. Değerlendir → confusion + accuracy vs SNR
9. Karşılaştırma grafiği

## Önemli notlar

**Colab oturumu kapanırsa ne olur?**
Drive'a kayıtlı her şey kalıcı: kod, processed dataset, checkpoint'ler, figürler. Sadece kurulu pip paketleri gider, onları her oturumda yeniden `pip install` etmen lazım (notebook'un 2. bölümü zaten yapıyor).

**Disk kotası**
Colab `/content/` ephemeral, ~80 GB. Drive ücretsiz tier 15 GB. Bu proje toplam ~5-6 GB tutuyor (raw pickle 640 MB + processed ~1.5 GB + checkpoint'ler birkaç MB), rahat sığar.

**GPU bitince**
Colab ücretsiz tier'da günde belli saatte GPU veriyor. Eğitim ortasında düşerse: en son kaydedilen `best.pt` Drive'da güvende, sonraki oturumda evaluate.py ile devam edebilirsin (ya da train'e resume eklenmemiş — gerekirse eklerim).

**`num_workers=0`**
Colab'da `num_workers > 0` bazen DataLoader'ı kilitliyor. Notebook bu yüzden `--num_workers 0` parametresiyle çağırıyor. Local makinede `0`'ı `2` yapabilirsin, hız artar.

**Pickle UnicodeDecodeError**
Olursa: pickle Python 2'de oluşturuldu. `data_loader.py` zaten `encoding='latin1'` ile yüklüyor, sorun olmamalı.

**Drive yolu farklıysa**
Notebook'taki `PROJECT_DIR = '/content/drive/MyDrive/amc_project'` satırını kendi yoluna göre düzenle.
