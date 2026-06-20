import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class PipelineRunner:
    def __init__(self, task, algorithm, params=None):
        self.task = task
        self.algorithm = algorithm
        self.params = params or {}
        self.model = None

    def _filter_params(self, algorithm):
        """Her algoritma için geçerli parametreleri filtrele"""
        algo_params = {
            'RandomForest': ['n_estimators', 'max_depth', 'min_samples_split',
                           'min_samples_leaf', 'max_features', 'random_state'],
            'XGBoost': ['n_estimators', 'max_depth', 'learning_rate',
                       'subsample', 'colsample_bytree', 'random_state'],
            'LightGBM': ['n_estimators', 'max_depth', 'learning_rate',
                        'num_leaves', 'subsample', 'random_state'],
            'LogisticRegression': ['C', 'max_iter', 'penalty', 'solver',
                                  'random_state']
        }
        allowed = algo_params.get(algorithm, [])
        return {k: v for k, v in self.params.items() if k in allowed}

    def _clean_data(self, X, y):
        """Veriyi temizle: NaN doldur, kategorikleri encode et, string sütunları çıkar"""
        # Sadece sayısal sütunları al
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in data. Cannot train model.")
        
        X_num = X[numeric_cols].copy()
        
        # NaN değerleri ortalama ile doldur
        if X_num.isnull().any().any():
            imputer = SimpleImputer(strategy='mean')
            X_num = pd.DataFrame(imputer.fit_transform(X_num), 
                                columns=numeric_cols, 
                                index=X_num.index)
        
        # Hedef değişkeni temizle
        y_clean = y.copy()
        
        # y'de NaN varsa o satırları sil
        if y_clean.isnull().any():
            mask = ~y_clean.isnull()
            X_num = X_num[mask]
            y_clean = y_clean[mask]
        
        # y kategorik/string ise encode et
        if y_clean.dtype == 'object' or y_clean.dtype == 'string':
            le = LabelEncoder()
            y_clean = pd.Series(le.fit_transform(y_clean.astype(str)), 
                              index=y_clean.index)
        
        return X_num, y_clean

    def run_tabular(self, df, target_col='target'):
        """Tabular veri ile model eğit"""
        # Hedef sütunu belirle
        if target_col not in df.columns:
            target_col = df.columns[-1]  # Son sütunu hedef al
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Veriyi temizle
        try:
            X_clean, y_clean = self._clean_data(X, y)
        except ValueError as e:
            return None, {"error": str(e)}
        
        # Eğitim/test bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Geçerli parametreleri al
        filtered_params = self._filter_params(self.algorithm)
        
        # Model seçimi
        if self.algorithm == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            self.model = (RandomForestClassifier(**filtered_params) 
                         if self.task == "classification" 
                         else RandomForestRegressor(**filtered_params))
                         
        elif self.algorithm == "XGBoost":
            from xgboost import XGBClassifier, XGBRegressor
            self.model = (XGBClassifier(**filtered_params) 
                         if self.task == "classification" 
                         else XGBRegressor(**filtered_params))
                         
        elif self.algorithm == "LightGBM":
            from lightgbm import LGBMClassifier, LGBMRegressor
            self.model = (LGBMClassifier(**filtered_params) 
                         if self.task == "classification" 
                         else LGBMRegressor(**filtered_params))
                         
        elif self.algorithm == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            # LogisticRegression özel parametreleri
            lr_params = {k: v for k, v in filtered_params.items() 
                        if k in ['C', 'max_iter', 'penalty', 'solver', 'random_state']}
            if 'max_iter' not in lr_params:
                lr_params['max_iter'] = 1000
            self.model = LogisticRegression(**lr_params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Eğit
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        
        # Sonuçları döndür
        if self.task == "classification":
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, zero_division=0)
            return self.model, {"accuracy": acc, "report": report}
        else:
            mse = mean_squared_error(y_test, preds)
            return self.model, {"MSE": mse}

    def run_sentiment(self, texts, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Sentiment analizi"""
        try:
            from transformers import pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
            results = sentiment_pipeline(texts)
            return results
        except ImportError:
            return [{"label": "ERROR", "score": 0.0, 
                    "error": "Install: pip install transformers"}]
        except Exception as e:
            return [{"label": "ERROR", "score": 0.0, "error": str(e)}]

    def run_yolo(self, image_paths, model_name="yolov8n.pt"):
        """YOLO object detection"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            all_results = []
            for img_path in image_paths:
                results = model(img_path)
                all_results.append(results)
            return all_results
        except ImportError:
            return [f"Error: Install ultralytics: pip install ultralytics"]
        except Exception as e:
            return [f"Error: {e}"]