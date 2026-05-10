# Türkçe Haber Metinleri Kategorizasyonu (Derin Öğrenme)

Bu proje, Derin Öğrenme dersi kapsamında "Bir problemi mühendislik bakış açısıyla çözmek" hedefiyle geliştirilmiştir. Projede, gerçek dünya problemi olan Türkçe metin sınıflandırma işlemi için **TTC-4900** veri seti kullanılmış ve iki farklı derin öğrenme mimarisi (1D-CNN ve Bi-LSTM + Masked Attention) karşılaştırmalı olarak analiz edilmiştir.

## 📌 Proje Özeti
* **Veri Seti:** TTC-4900 (HuggingFace üzerinden `savasy/ttc4900`)
* **Sınıflar:** Siyaset, Dünya, Ekonomi, Kültür, Sağlık, Spor, Teknoloji
* **Modeller:** 
  1. Paralel Çok-Kernelli 1D-CNN (TextCNN)
  2. Çift Yönlü LSTM ve Masked Self-Attention (Bi-LSTM + Attention)
* **Değerlendirme Metrikleri:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## 🚀 Kurulum ve Gereksinimler

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/DerinOgrenmeProjesi.git](https://github.com/KULLANICI_ADINIZ/DerinOgrenmeProjesi.git)
   cd DerinOgrenmeProjesi