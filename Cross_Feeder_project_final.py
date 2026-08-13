import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class AnomalyDetectionPipeline:
    """Telekom Hücre (Cell) ve Handover (HO) Verileri İçin Etiketsiz (Unsupervised) Isolation Forest Pipeline."""

    def __init__(self, target_col: Optional[str] = None):
        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None
        self.df_cleaned: Optional[pd.DataFrame] = None
        self.df_transformed: Optional[pd.DataFrame] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None

        self.scaler: Optional[StandardScaler] = None
        self.best_model: Optional[IsolationForest] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.evaluation_metrics: Optional[Dict[str, Any]] = None

    def load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        """Veritabanı ve merge işlemleri yerine hazır CSV dosyasını okur."""
        print("\n" + "=" * 60)
        print("1. ADIM: CSV DOSYASINDAN HAZIR VERİNİN ALINMASI")
        print("=" * 60)

        try:
            self.df = pd.read_csv(filepath)
            self.df = self.df.drop("RW", axis=1)
            # Sütun isimlerini standart olması için büyük harfe çevirelim
            self.df.columns = [c.upper() for c in self.df.columns]

            print(f"✅ CSV dosyası başarıyla yüklendi: {filepath}")
            print(f"✅ Boyut: {len(self.df)} satır x {self.df.shape[1]} sütun")
            print(f"✅ İlk 5 satır: \n{self.df.head()}")
            return self.df

        except Exception as e:
            print(f"❌ CSV Yükleme Hatası: {e}")
            raise e

    def inspect_data(self) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("2. ADIM: VERİ ÖN İNCELEME")
        print("=" * 60)

        rows, cols = self.df.shape
        print(f"-> Boyutlar: {rows} Satır x {cols} Sütun")

        info_df = pd.DataFrame(
            {
                "Veri Tipi": self.df.dtypes,
                "Eksik Sayısı": self.df.isnull().sum(),
                "Eksik (%)": (self.df.isnull().sum() / rows * 100).round(2),
            }
        )
        print("\n-> Sütunlar ve Eksik Veri Durumu:")
        print(info_df.to_string())

        return {"shape": (rows, cols), "info": info_df}

    def preprocess_data(
        self, logical_rules: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("3. ADIM: VERİ TEMİZLEME VE ÖN İŞLEME")
        print("=" * 60)

        self.df_cleaned = self.df.copy()

        dups = self.df_cleaned.duplicated().sum()
        if dups > 0:
            self.df_cleaned = self.df_cleaned.drop_duplicates().reset_index(
                drop=True
            )
            print(f"[TEMİZLEME] {dups} mükerrer satır silindi.")

        for col in self.df_cleaned.columns:
            if self.df_cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.df_cleaned[col]):
                    med = self.df_cleaned[col].median()
                    self.df_cleaned[col] = self.df_cleaned[col].fillna(med)
                    print(
                        f"[TEMİZLEME] '{col}' (Numeric) Medyan ({med}) ile dolduruldu."
                    )
                else:
                    mod = self.df_cleaned[col].mode()[0]
                    self.df_cleaned[col] = self.df_cleaned[col].fillna(mod)
                    print(
                        f"[TEMİZLEME] '{col}' (Kategorik) Mod ('{mod}') ile dolduruldu."
                    )

        if logical_rules:
            for r_name, r_query in logical_rules.items():
                before = len(self.df_cleaned)
                self.df_cleaned = self.df_cleaned.query(r_query).reset_index(
                    drop=True
                )
                print(
                    f"[TEMİZLEME] Kural '{r_name}': {before - len(self.df_cleaned)} satır çıkarıldı."
                )

        if "HO_ATTEMPTS" in self.df_cleaned.columns:
            q1 = self.df_cleaned["HO_ATTEMPTS"].quantile(0.25)
            q3 = self.df_cleaned["HO_ATTEMPTS"].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            outliers_count = (
                (self.df_cleaned["HO_ATTEMPTS"] < lower)
                | (self.df_cleaned["HO_ATTEMPTS"] > upper)
            ).sum()
            if outliers_count > 0:
                self.df_cleaned["HO_ATTEMPTS"] = np.where(
                    self.df_cleaned["HO_ATTEMPTS"] < lower,
                    lower,
                    self.df_cleaned["HO_ATTEMPTS"],
                )
                self.df_cleaned["HO_ATTEMPTS"] = np.where(
                    self.df_cleaned["HO_ATTEMPTS"] > upper,
                    upper,
                    self.df_cleaned["HO_ATTEMPTS"],
                )
                print(
                    f"[TEMİZLEME] 'HO_ATTEMPTS' için {outliers_count} aykırı değer IQR sınırlarına baskılandı."
                )

        return self.df_cleaned

    def apply_feature_engineering(self, custom_creation_func: Optional[callable] = None) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("4. ADIM: FEATURE ENGINEERING")
        print("=" * 60)

        self.df_transformed = self.df_cleaned.copy()

        if custom_creation_func:
            self.df_transformed = custom_creation_func(self.df_transformed)
        else:
            if "SRC_AZIMUTH" in self.df_transformed.columns and "TGT_AZIMUTH" in self.df_transformed.columns:
                self.df_transformed["AZIMUTH_DIFF"] = np.abs(
                    self.df_transformed["SRC_AZIMUTH"] - self.df_transformed["TGT_AZIMUTH"]
                )
                print("[FEATURE] 'AZIMUTH_DIFF' (Azimut Farkı) değişkeni türetildi.")

            if all(col in self.df_transformed.columns for col in ["SRC_LATITUDE", "SRC_LONGITUDE", "TGT_LATITUDE", "TGT_LONGITUDE"]):
                R = 6371.0
                lat1, lon1 = np.radians(self.df_transformed["SRC_LATITUDE"]), np.radians(self.df_transformed["SRC_LONGITUDE"])
                lat2, lon2 = np.radians(self.df_transformed["TGT_LATITUDE"]), np.radians(self.df_transformed["TGT_LONGITUDE"])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                
                self.df_transformed["DISTANCE_KM"] = R * c
                print("[FEATURE] 'DISTANCE_KM' (Hücreler Arası Mesafe) değişkeni türetildi.")

        # Sadece gereksiz SITE ID'leri atalım, CELL NAME'ler kalsın!
        non_feature_cols = ["SRC_SITE_ID", "TGT_SITE_ID"]
        cols_to_drop = [c for c in non_feature_cols if c in self.df_transformed.columns]
        
        if cols_to_drop:
            self.df_transformed = self.df_transformed.drop(columns=cols_to_drop)
            print(f"[FEATURE] Gereksiz ID sütunları çıkarıldı: {cols_to_drop}")

        return self.df_transformed

    def split_data(self, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42, ):
        print("\n" + "=" * 60)
        print("5. ADIM: TRAIN / VALIDATION / TEST SPLIT & SCALING")
        print("=" * 60)

        X = self.df_transformed.copy()
        val_test_sum = val_ratio + test_ratio
        X_train, X_temp = train_test_split(X, test_size=val_test_sum, random_state=random_state)

        if test_ratio > 0:
            rel_test = test_ratio / val_test_sum
            X_val, X_test = train_test_split(X_temp, test_size=rel_test, random_state=random_state)
        else:
            X_val = X_temp
            X_test = None

        # Standard Scaler uygulanacak sütunlar listesinden koordinatları çıkarın
        exclude_cols = [ 'SRC_CELL_LATITUDE', 'SRC_CELL_LONGITUDE', 'SRC_CELL_AZIMUTH', 'TRG_CELL_LATITUDE', 'TRG_CELL_LONGITUDE', 'TRG_CELL_AZIMUTH' ]

        # Sadece gerçekten ölçeklenmesi gereken sayısal sütunları seçin
        num_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
        print(f"Numeric Columns  : {num_cols}")
            
        self.scaler = StandardScaler()
        X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = self.scaler.transform(X_val[num_cols])
        if X_test is not None:
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        print(f"-> Train Seti      : {len(X_train)} satır (%{len(X_train)/len(X)*100:.1f})")
        print(f"-> Validation Seti : {len(X_val)} satır (%{len(X_val)/len(X)*100:.1f})")
        if X_test is not None:
            print(f"-> Test Seti       : {len(X_test)} satır (%{len(X_test)/len(X)*100:.1f})")

    def tune_and_train(self, param_grid: Optional[Dict[str, List[Any]]] = None):
        print("\n" + "=" * 60)
        print("6 & 7. ADIM: ISOLATION FOREST MODEL EĞİTİMİ VE TUNING")
        print("=" * 60)

        if param_grid is None:
            param_grid = {
                "contamination": [0.03],
                "n_estimators": [100, 200],
                "max_samples": ["auto", 256],
            }

        # Modeli SADECE sayısal sütunlarla eğitiyoruz!
        X_train_num = self.X_train.select_dtypes(include=[np.number])
        X_val_num = self.X_val.select_dtypes(include=[np.number])

        best_score = -float("inf")
        best_params = None
        best_model = None

        for cont in param_grid.get("contamination", [0.03]):
            for n_est in param_grid.get("n_estimators", [100]):
                for m_samp in param_grid.get("max_samples", ["auto"]):

                    model = IsolationForest(
                        contamination=cont,
                        n_estimators=n_est,
                        max_samples=m_samp,
                        random_state=42,
                        n_jobs=-1,
                    )
                    model.fit(X_train_num)

                    val_preds = model.predict(X_val_num)
                    val_scores = model.decision_function(X_val_num)

                    score = val_scores[val_preds == 1].mean() - val_scores[val_preds == -1].mean() if (-1 in val_preds) else -1.0

                    if score > best_score:
                        best_score = score
                        best_params = {"contamination": cont, "n_estimators": n_est, "max_samples": m_samp}
                        best_model = model

        self.best_model = best_model
        self.best_params = best_params

        print(f"[MODEL EĞİTİLDİ] Seçilen Hiperparametreler:")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")

        return self.best_model, self.best_params

    def evaluate_test(self, top_n: int = 10) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("8. ADIM: TEST SETİ İLE DEĞERLENDİRME")
        print("=" * 60)

        eval_X = self.X_test if self.X_test is not None else self.X_val
        eval_X_num = eval_X.select_dtypes(include=[np.number])

        preds = self.best_model.predict(eval_X_num)
        scores = self.best_model.decision_function(eval_X_num)

        results_df = eval_X.copy()
        results_df["anomaly_label"] = preds
        results_df["anomaly_score"] = scores

        anomalies_cnt = (preds == -1).sum()
        total_cnt = len(eval_X)
        anomaly_ratio = (anomalies_cnt / total_cnt) * 100

        print(f"-> Tespit Edilen Anomali Sayısı: {anomalies_cnt} / {total_cnt} (%{anomaly_ratio:.2f})")
        print(f"-> Ortalama Anomali Skoru    : {scores.mean():.4f}")

        anomalies_df = results_df[results_df["anomaly_label"] == -1].sort_values(by="anomaly_score", ascending=True)

        # Görsellikte Hücre İsimlerini en sol tarafa taşımak için sütun sırasını düzenliyoruz
        name_cols = [c for c in ["SRC_CELL_NAME", "TGT_CELL_NAME"] if c in anomalies_df.columns]
        other_cols = [c for c in anomalies_df.columns if c not in name_cols]
        anomalies_df = anomalies_df[name_cols + other_cols]

        print("\n" + "-" * 50)
        print(f"🚨 TESPİT EDİLEN EN BELİRGİN İLK {min(top_n, len(anomalies_df))} ANOMALİ")
        print("-" * 50)

        if not anomalies_df.empty:
            with pd.option_context("display.max_columns", None, "display.width", 1000):
                print(anomalies_df.head(top_n).to_string(index=False)) # index=False diyerek kafa karıştıran sayıları da kaldırıyoruz
        else:
            print("✅ Herhangi bir anomali tespit edilmedi.")
        print("-" * 50)

        metrics = {
            "results_df": results_df,
            "anomalies_df": anomalies_df,
            "anomalies_cnt": anomalies_cnt,
            "anomaly_ratio": anomaly_ratio,
        }

        self.evaluation_metrics = metrics
        return metrics
    
    def visualize(self):
        print("\n" + "=" * 60)
        print("9. ADIM: ANOMALİ GÖRSELLEŞTİRME (t-SNE İLE)")
        print("=" * 60)

        results_df = self.evaluation_metrics["results_df"]
        eval_X = self.X_test if self.X_test is not None else self.X_val

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.set_theme(style="whitegrid")

        # 1. Anomali Skor Dağılımı
        sns.histplot(
            data=results_df,
            x="anomaly_score",
            hue="anomaly_label",
            palette={1: "royalblue", -1: "crimson"},
            kde=True,
            bins=30,
            ax=axes[0],
        )
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1.5)
        axes[0].set_title("Anomali Skor Dağılımı")

        # 2. t-SNE 2D İndirgeme
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_coords = tsne.fit_transform(
            eval_X.select_dtypes(include=[np.number])
        )
        tsne_df = pd.DataFrame(tsne_coords, columns=["t-SNE1", "t-SNE2"])
        tsne_df["anomaly_label"] = results_df["anomaly_label"].values

        sns.scatterplot(
            data=tsne_df,
            x="t-SNE1",
            y="t-SNE2",
            hue="anomaly_label",
            palette={1: "royalblue", -1: "crimson"},
            style="anomaly_label",
            markers={1: "o", -1: "X"},
            s=60,
            alpha=0.8,
            ax=axes[1],
        )
        axes[1].set_title("2D t-SNE Anomali Dağılımı")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        # Dosya adını projenize göre ayarlayabilirsiniz
        csv_file_path = "export_cross_feeder_train_data_set.csv"

        pipeline = AnomalyDetectionPipeline(target_col=None)

        # 1. Doğrudan CSV dosyasını yüklüyoruz
        pipeline.load_data_from_csv(csv_file_path)

        # 2. Diğer pipeline adımları aynen çalışmaya devam eder
        pipeline.inspect_data()
        pipeline.preprocess_data()
        pipeline.apply_feature_engineering()
        pipeline.split_data(val_ratio=0.15, test_ratio=0.15)

        grid = {
            "contamination": [0.03],
            "n_estimators": [100, 200],
            "max_samples": ["auto", 256],
        }
        pipeline.tune_and_train(param_grid=grid)

        pipeline.evaluate_test()
        pipeline.visualize()

    except Exception as e:
        print(f"❌ Akış Hatası: {e}")