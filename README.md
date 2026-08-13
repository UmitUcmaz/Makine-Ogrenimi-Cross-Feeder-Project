# 📡 Telecom Anomaly Detection Pipeline (Isolation Forest)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Isolation%20Forest-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)

Telekomünikasyon ağlarındaki Hücre (Cell) ve Handover (HO) verileri üzerinde **etiketsiz (unsupervised)** anomali tespiti yapan nesne yönelimli (OOP) makine öğrenmesi pipeline'ı. 

Bu proje; fiziki konumlar, yönelim açıları ve geçiş istatistiklerine dayanarak hatalı yapılandırılmış veya aykırı davranış sergileyen hücre çiftlerini tespit etmeyi amaçlar.

---

## 📌 Öne Çıkan Özellikler

* **Veri Yükleme & Ön İşleme:** CSV/Veritabanı kaynaklı verileri işleme, mükerrer satırları temizleme, eksik verileri medyan/mod ile doldurma ve aykırı değerleri IQR ile baskılama.
* **Geometrik & Açısal Öznitelik Mühendisliği (Feature Engineering):**
  * **Haversine Mesafesi (`DISTANCE_KM`):** Enlem ve boylamlardan ($3D$ küresel geometry) kuş uçuşu hücresel mesafe hesaplama.
  * **Azimut Farkı (`AZIMUTH_DIFF`):** Hücrelerin yönelim açıları arasındaki mutlak farkı türetme.
* **Akıllı Scaling:** Koordinat/açı verilerini bozulmaya uğratmamak adına `StandardScaler` uygulamasından muaf tutan/doğrudan türetilmiş metrikleri ölçekleyen yapı.
* **Isolation Forest & Grid Search:** En uygun kirlilik (`contamination`) ve hiperparametre kümesini validation seti üzerinde bularak model eğitimi.
* **Görselleştirme & Analiz:**
  * **Anomali Skor Dağılımı:** Decision boundary etrafındaki ayrışmayı gösteren histogram/KDE grafiği.
  * **2D t-SNE Kümeleme:** Yüksek boyutlu veriyi 2 boyuta indirgeyerek izole anomalileri uzamsal gösterim.
  * **Açıklanabilir Raporlama:** En belirgin anomalileri ilgili Hücre Adları (`SRC_CELL_NAME`, `TGT_CELL_NAME`) ve skorlarıyla birlikte terminale/rapora dökme.

---

## 📁 Proje Yapısı

```text
.
├── dataset.csv                   # Kullanılan veri seti (Varsayılan)
├── anomaly_pipeline.py           # AnomalyDetectionPipeline sınıfı ve ana akış
├── README.md                     # Proje dokümantasyonu
└── requirements.txt              # Gerekli Python kütüphaneleri
```
