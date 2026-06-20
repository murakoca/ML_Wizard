from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QCheckBox,
                             QFileDialog, QMessageBox)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Stage10Interpretability(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Model Interpretability & Ethics")
        self.set_theory("<h2>10. Model Interpretability & Ethics</h2><p>SHAP, LIME, Global vs Local Explanations, Fairness Metrics, Data Privacy, Adversarial Robustness, Guardrails.</p>")
        self.df = None
        self.model = None
        self.X_test = None
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Veri yükleme
        data_gb = QGroupBox("1. Model & Data")
        data_vbox = QVBoxLayout()
        btn_load = QPushButton("Load CSV (from previous stages)")
        btn_load.clicked.connect(self.load_csv)
        data_vbox.addWidget(btn_load)

        self.data_info = QLabel("No model loaded")
        data_vbox.addWidget(self.data_info)

        btn_train = QPushButton("Quick Train Random Forest (for demo)")
        btn_train.clicked.connect(self.quick_train)
        data_vbox.addWidget(btn_train)
        data_gb.setLayout(data_vbox)
        self.layout.addWidget(data_gb)

        # Yorumlanabilirlik metodu
        interp_gb = QGroupBox("2. Interpretability Method")
        interp_vbox = QVBoxLayout()
        self.method_cb = QComboBox()
        self.method_cb.addItems(["SHAP", "LIME", "Feature Importance", "Partial Dependence"])
        interp_vbox.addWidget(QLabel("Method:"))
        interp_vbox.addWidget(self.method_cb)
        interp_gb.setLayout(interp_vbox)
        self.layout.addWidget(interp_gb)

        # Etik & Fairness
        ethics_gb = QGroupBox("3. Ethics & Fairness")
        ethics_vbox = QVBoxLayout()
        self.fairness_checks = {}
        for metric in ["Demographic Parity", "Equal Opportunity", "Disparate Impact",
                       "Data Privacy (DP)", "Adversarial Robustness"]:
            cb = QCheckBox(metric)
            self.fairness_checks[metric] = cb
            ethics_vbox.addWidget(cb)
        ethics_gb.setLayout(ethics_vbox)
        self.layout.addWidget(ethics_gb)

        # Run
        self.run_btn = QPushButton("4. Analyze Model")
        self.run_btn.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.data_info.setText(f"Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} cols")
            self.settings.update("interpretability", "data_path", path)

    def quick_train(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Load CSV first.")
            return
        try:
            target_col = self.df.columns[-1]
            X = self.df.drop(columns=[target_col]).select_dtypes(include=['number'])
            y = self.df[target_col]
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            acc = self.model.score(X_test, y_test)
            self.data_info.setText(f"Model trained! Accuracy: {acc:.4f} | Test samples: {len(X_test)}")
        except Exception as e:
            self.data_info.setText(f"Error: {e}")

    def run_analysis(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        method = self.method_cb.currentText()
        selected_fairness = [m for m, cb in self.fairness_checks.items() if cb.isChecked()]

        self.output.clear()
        self.output.append(f"🔍 Model Analysis Report\n{'='*50}")

        # Feature Importance (her zaman çalışır)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            self.output.append(f"\n📊 Top 10 Feature Importances:")
            feature_names = self.X_test.columns if hasattr(self.X_test, 'columns') else [f"F{i}" for i in range(len(importances))]
            for i, idx in enumerate(indices[:10]):
                self.output.append(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

        # Metod spesifik açıklamalar
        if method == "SHAP":
            self.output.append(f"\n🍏 SHAP (SHapley Additive exPlanations)")
            self.output.append(f"- Would compute Shapley values for each feature")
            self.output.append(f"- Global bar plot + Local waterfall plot")
            self.output.append(f"- Install: pip install shap")
            self.output.append(f"- Usage: shap.Explainer(model).shap_values(X_test)")

        elif method == "LIME":
            self.output.append(f"\n🍋 LIME (Local Interpretable Model-agnostic Explanations)")
            self.output.append(f"- Creates local surrogate models")
            self.output.append(f"- Explains individual predictions")
            self.output.append(f"- Install: pip install lime")
            self.output.append(f"- Usage: LimeTabularExplainer().explain_instance()")

        elif method == "Partial Dependence":
            self.output.append(f"\n📈 Partial Dependence Plots")
            self.output.append(f"- Shows marginal effect of features on predictions")
            self.output.append(f"- From sklearn.inspection import PartialDependenceDisplay")

        # Fairness raporu
        if selected_fairness:
            self.output.append(f"\n⚖️ Fairness & Ethics Analysis")
            for metric in selected_fairness:
                if metric == "Demographic Parity":
                    self.output.append(f"  ✅ Demographic Parity: P(ŷ=1|A=a) = P(ŷ=1|A=b)")
                    self.output.append(f"     Install: pip install fairlearn")
                elif metric == "Equal Opportunity":
                    self.output.append(f"  ✅ Equal Opportunity: Equal TPR across groups")
                elif metric == "Disparate Impact":
                    self.output.append(f"  ✅ Disparate Impact Ratio should be > 0.8")
                    self.output.append(f"     Reference: US EEOC 80% rule")
                elif metric == "Data Privacy":
                    self.output.append(f"  🔒 Differential Privacy (ε-differential privacy)")
                    self.output.append(f"     Install: pip install opacus (for PyTorch)")
                    self.output.append(f"     Install: pip install diffprivlib (for sklearn)")
                elif metric == "Adversarial Robustness":
                    self.output.append(f"  🛡️ Would test against adversarial attacks")
                    self.output.append(f"     - FGSM, PGD attacks")
                    self.output.append(f"     - Install: pip install adversarial-robustness-toolbox")

        # Guardrails önerisi
        self.output.append(f"\n🛡️ AI Safety Guardrails")
        self.output.append(f"  - Jailbreak detection")
        self.output.append(f"  - Prompt injection prevention")
        self.output.append(f"  - Output validation")
        self.output.append(f"  - Install: pip install guardrails-ai")
        self.output.append(f"  - NVIDIA NeMo Guardrails also available")

        self.settings.update("interpretability", "method", method)
        self.settings.update("interpretability", "fairness_checks", selected_fairness)