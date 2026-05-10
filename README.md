# 🗞️ Türkçe Haber Metinleri Kategorizasyonu

> Derin Öğrenme teknikleriyle Türkçe haber metinlerinin otomatik sınıflandırılması — **1D-CNN** ve **Bi-LSTM + Masked Attention** mimarilerinin karşılaştırmalı analizi.

---

## 📌 Proje Özeti

| Özellik | Detay |
|---|---|
| **Veri Seti** | [TTC-4900](https://huggingface.co/datasets/savasy/ttc4900) (`savasy/ttc4900`) |
| **Sınıf Sayısı** | 7 (Siyaset, Dünya, Ekonomi, Kültür, Sağlık, Spor, Teknoloji) |
| **Mimari 1** | Paralel Çok-Kernelli 1D-CNN (TextCNN) |
| **Mimari 2** | Çift Yönlü LSTM + Masked Self-Attention |
| **Değerlendirme** | Accuracy · Precision · Recall · F1-Score · ROC-AUC · Confusion Matrix |

---

## 🚀 Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/KULLANICI_ADINIZ/DerinOgrenmeProjesi.git
cd DerinOgrenmeProjesi
```

### 2. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Eğitimi

Ana dosyayı çalıştırarak veri işleme, eğitim ve değerlendirme adımlarını başlatın.  
Kod; **Apple Silicon (MPS)**, **CUDA** ve **CPU**'yu otomatik algılayarak uygun donanım hızlandırması ile çalışır.

```bash
python main.py
```

Bu komut sırasıyla şu işlemleri gerçekleştirir:

1. Veri setini indirir ve **Train / Val / Test** (%70 · %15 · %15) olarak böler
2. Keşifsel Veri Analizi (EDA) grafiklerini `plots/` klasörüne kaydeder
3. Özel kelime dağarcığını (Vocab) oluşturur ve `vocab.json` olarak kaydeder
4. Öğrenme hızı (Learning Rate) duyarlılık analizi yapar
5. **TextCNN** ve **Bi-LSTM** modellerini eğitir, en iyi ağırlıkları `.pt` formatında saklar
6. Karmaşıklık matrisleri, öğrenme eğrileri ve karşılaştırma grafiklerini üretir

---

## 🔮 Tahmin (Inference)

Eğitilmiş model ağırlıklarıyla yeni haber metinleri üzerinde tahmin yapabilirsiniz.  
Varsayılan olarak en yüksek doğruluğa sahip **Bi-LSTM + Attention** modeli kullanılır.

**İnteraktif mod** — sürekli metin girişi için:

```bash
python predict.py
```

**Tek seferlik tahmin** — metni doğrudan argüman olarak verin:

```bash
python predict.py "Merkez Bankası'nın faiz kararını piyasa beklentileri doğrultusunda sabit bırakmasının ardından, Borsa İstanbul (BIST 100) endeksi günü rekor seviyede kapattı."
```

---

## 📁 Proje Yapısı

```
DerinOgrenmeProjesi/
│
├── main.py                   # Veri işleme, model mimarileri, eğitim ve grafik çizimleri
├── predict.py                # Eğitilmiş model ile yeni metin tahmini
├── requirements.txt          # Gerekli Python kütüphaneleri
│
├── vocab.json                # Eğitim sırasında oluşturulan kelime sözlüğü  ¹
├── best_TextCNN.pt           # Eğitilmiş TextCNN model ağırlıkları  ¹
├── best_Bi-LSTM_Attention.pt # Eğitilmiş Bi-LSTM model ağırlıkları  ¹
│
└── plots/                    # Öğrenme eğrileri, karmaşıklık matrisleri, EDA grafikleri  ¹
```

> ¹ Bu dosya ve klasörler eğitim tamamlandıktan sonra otomatik olarak oluşturulur.

---

## 📊 Değerlendirme Metrikleri

Modeller aşağıdaki metrikler üzerinden kıyaslanmıştır:

- **Accuracy** — Genel doğruluk oranı
- **Precision / Recall / F1-Score** — Sınıf bazlı denge analizi
- **ROC-AUC** — Ayrıştırma performansı
- **Confusion Matrix** — Sınıflar arası hata dağılımı

---

<details>
<summary><strong>⚙️ Sistem Gereksinimleri</strong></summary>

- Python 3.8+
- PyTorch ≥ 2.0 (MPS / CUDA / CPU destekli)
- Tüm bağımlılıklar `requirements.txt` içinde tanımlıdır

</details>