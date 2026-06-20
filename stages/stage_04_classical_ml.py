from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QPushButton,
                             QTextEdit, QComboBox, QFileDialog, QMessageBox)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

class Stage04ClassicalML(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Classical Machine Learning")
        self.set_theory("<h2>04. Classical Machine Learning</h2><p>Regression, Classification, Loss Functions, Regularization, Bias-Variance Tradeoff, Cross Validation, Clustering, Ensemble Methods.</p>")
        self.df = None
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Veri yükleme
        load_gb = QGroupBox("1. Load Data (or use from previous stage)")
        load_vbox = QVBoxLayout()
        btn_load = QPushButton("Load CSV")
        btn_load.clicked.connect(self.load_csv)
        load_vbox.addWidget(btn_load)
        self.load_info = QLabel("No data. Go back to Stage 3 first, or load here.")
        load_vbox.addWidget(self.load_info)
        load_gb.setLayout(load_vbox)
        self.layout.addWidget(load_gb)

        # Algoritma seçimi
        algo_gb = QGroupBox("2. Select Algorithm")
        algo_vbox = QVBoxLayout()
        algo_vbox.addWidget(QLabel("Task:"))
        self.task_cb = QComboBox()
        self.task_cb.addItems(["classification", "regression"])
        self.task_cb.currentTextChanged.connect(self._update_algorithms)
        algo_vbox.addWidget(self.task_cb)

        algo_vbox.addWidget(QLabel("Algorithm:"))
        self.algo_cb = QComboBox()
        self._update_algorithms("classification")
        algo_vbox.addWidget(self.algo_cb)
        algo_gb.setLayout(algo_vbox)
        self.layout.addWidget(algo_gb)

        # Run butonu
        self.run_btn = QPushButton("3. Run Model")
        self.run_btn.clicked.connect(self.run_model)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def _update_algorithms(self, task):
        self.algo_cb.clear()
        if task == "classification":
            self.algo_cb.addItems(["Logistic Regression", "Decision Tree", "Random Forest"])
        else:
            self.algo_cb.addItems(["Linear Regression", "Decision Tree", "Random Forest"])

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.load_info.setText(f"Loaded: {path} | Shape: {self.df.shape}")
            self.settings.update("classical_ml", "data_path", path)

    def run_model(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load a CSV file first.")
            return
        try:
            # Son sütunu hedef kabul et
            target_col = self.df.columns[-1]
            X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
            y = self.df[target_col]

            # Hedef kategorik ise encode
            if self.task_cb.currentText() == "classification" and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            algo = self.algo_cb.currentText()
            if algo == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo == "Linear Regression":
                model = LinearRegression()
            elif algo == "Decision Tree":
                model = DecisionTreeClassifier() if self.task_cb.currentText()=="classification" else DecisionTreeRegressor()
            elif algo == "Random Forest":
                model = RandomForestClassifier(n_estimators=100) if self.task_cb.currentText()=="classification" else RandomForestRegressor(n_estimators=100)
            else:
                self.output.setText("Unknown algorithm")
                return

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if self.task_cb.currentText() == "classification":
                acc = accuracy_score(y_test, preds)
                self.output.setText(f"✅ {algo}\nAccuracy: {acc:.4f}")
            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                self.output.setText(f"✅ {algo}\nMSE: {mse:.4f}\nR²: {r2:.4f}")

            self.settings.update("classical_ml", "algorithm", algo)
            self.settings.update("classical_ml", "task", self.task_cb.currentText())
        except Exception as e:
            self.output.setText(f"❌ Error: {e}")