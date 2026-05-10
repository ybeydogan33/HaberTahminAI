# Türkçe Haber Metinleri Kategorizasyonu (Derin Öğrenme)

Bu proje, Derin Öğrenme dersi kapsamında "Bir problemi mühendislik bakış açısıyla çözmek" hedefiyle geliştirilmiştir. Projede, gerçek dünya problemi olan Türkçe metin sınıflandırma işlemi için **TTC-4900** veri seti kullanılmış ve iki farklı derin öğrenme mimarisi (1D-CNN ve Bi-LSTM + Masked Attention) karşılaştırmalı olarak analiz edilmiştir.

## 📌 Proje Özeti
* **Veri Seti:** TTC-4900 (HuggingFace üzerinden `savasy/ttc4900`)
* **Sınıflar (7 Adet):** Siyaset, Dünya, Ekonomi, Kültür, Sağlık, Spor, Teknoloji
* **Modeller:** 
  1. Paralel Çok-Kernelli 1D-CNN (TextCNN)
  2. Çift Yönlü LSTM ve Masked Self-Attention (Bi-LSTM + Attention)
* **Değerlendirme Metrikleri:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## 🚀 Kurulum ve Gereksinimler

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın ve proje dizinine gidin:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/DerinOgrenmeProjesi.git](https://github.com/KULLANICI_ADINIZ/DerinOgrenmeProjesi.git)
   cd DerinOgrenmeProjesi

```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt


```



```

## 🧠 Modelleri Eğitme (Training)

Sıfırdan eğitim yapmak, veri analizi (EDA) grafiklerini çıkarmak ve modelleri karşılaştırmak için ana dosyayı çalıştırın. Kod; Apple Silicon (MPS), CUDA veya CPU'yu otomatik algılayarak donanım hızlandırması ile eğitimi başlatacaktır.

```bash
python main.py

```

Bu komut sırasıyla şunları gerçekleştirir:

* Veri setini indirir ve Train/Val/Test (%70, %15, %15) olarak böler.
* Sınıf dağılımı ve metin uzunluğu gibi Keşifsel Veri Analizi (EDA) grafiklerini `plots/` klasörüne kaydeder.
* Özel kelime dağarcığını (Vocab) oluşturur ve `vocab.json` olarak kaydeder.
* Öğrenme hızı (Learning Rate) duyarlılık analizi yapar.
* TextCNN ve Bi-LSTM modellerini eğitir, en iyi ağırlıkları (`.pt` formatında) kaydeder.
* Karmaşıklık matrisleri, öğrenme eğrileri ve model karşılaştırma grafiklerini oluşturur.

## 🔮 Tahmin Yapma (Inference)

Eğitilmiş model ağırlıklarını kullanarak yeni haber metinlerinin kategorisini tahmin etmek için `predict.py` dosyasını kullanabilirsiniz. (Varsayılan olarak en yüksek doğruluğa sahip Bi-LSTM + Attention modeli kullanılmaktadır).

**İnteraktif Mod (Sürekli metin girişi için):**

```bash
python predict.py

```

**Tek Seferlik Tahmin (Terminalden doğrudan metin vererek):**

```bash
python predict.py "Merkez Bankası'nın faiz kararını piyasa beklentileri doğrultusunda sabit bırakmasının ardından, Borsa İstanbul (BIST 100) endeksi günü rekor seviyede kapattı."

```

## 📁 Proje Yapısı

* `main.py`: Veri işleme, model mimarileri, eğitim döngüsü, hiperparametre analizi ve grafik çizimlerini içeren ana dosya.
* `predict.py`: Eğitilmiş model ile yeni metinlerin tahmin edilmesini sağlayan çıkarım dosyası.
* `requirements.txt`: Projenin çalışması için gerekli Python kütüphanelerinin listesi.
* `vocab.json`: Eğitim sırasında oluşturulan ve tahminleme için gerekli olan kelime sözlüğü dosyası. (Eğitim sonrası oluşur)
* `best_TextCNN.pt`: Eğitilmiş TextCNN model ağırlıkları. (Eğitim sonrası oluşur)
* `best_Bi-LSTM_Attention.pt`: Eğitilmiş Bi-LSTM model ağırlıkları. (Eğitim sonrası oluşur)
* `plots/`: Öğrenme eğrileri, karmaşıklık matrisleri ve veri analizi grafiklerini barındıran klasör. (Eğitim sonrası otomatik oluşur)
