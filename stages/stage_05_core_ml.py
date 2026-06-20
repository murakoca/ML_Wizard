from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QSpinBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox)
import pandas as pd
import numpy as np
from utils.synthetic_data import SyntheticDataGenerator
from utils.pipeline_runner import PipelineRunner

class Stage05CoreML(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Core Machine Learning")
        self.set_theory("<h2>05. Core ML (Advanced Algorithms)</h2><p>Regression, Classification, Loss Functions, Regularization, Cross Validation, Clustering, Ensemble Methods (XGBoost, LightGBM).</p>")
        self.df = None
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Veri
        data_gb = QGroupBox("1. Data Source")
        data_vbox = QVBoxLayout()
        btn_load = QPushButton("Load CSV File")
        btn_load.clicked.connect(self.load_csv)
        data_vbox.addWidget(btn_load)

        synth_row = QHBoxLayout()
        self.synth_btn = QPushButton("Generate Synthetic Data")
        self.synth_btn.clicked.connect(self.generate_synthetic)
        synth_row.addWidget(self.synth_btn)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 50000)
        self.samples_spin.setValue(1000)
        synth_row.addWidget(QLabel("Samples:"))
        synth_row.addWidget(self.samples_spin)
        data_vbox.addLayout(synth_row)

        self.data_status = QLabel("No data loaded")
        data_vbox.addWidget(self.data_status)
        data_gb.setLayout(data_vbox)
        self.layout.addWidget(data_gb)

        # Algoritma + hiperparametre
        algo_gb = QGroupBox("2. Algorithm & Hyperparameters")
        algo_vbox = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Task:"))
        self.task_cb = QComboBox()
        self.task_cb.addItems(["classification", "regression"])
        row1.addWidget(self.task_cb)
        row1.addWidget(QLabel("Algorithm:"))
        self.algo_cb = QComboBox()
        self.algo_cb.addItems(["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"])
        row1.addWidget(self.algo_cb)
        algo_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("n_estimators:"))
        self.n_est = QSpinBox()
        self.n_est.setRange(10, 1000)
        self.n_est.setValue(100)
        row2.addWidget(self.n_est)
        row2.addWidget(QLabel("max_depth:"))
        self.max_depth = QSpinBox()
        self.max_depth.setRange(1, 50)
        self.max_depth.setValue(6)
        row2.addWidget(self.max_depth)
        row2.addWidget(QLabel("learning_rate:"))
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.001, 1.0)
        self.lr.setSingleStep(0.01)
        self.lr.setValue(0.1)
        row2.addWidget(self.lr)
        algo_vbox.addLayout(row2)
        algo_gb.setLayout(algo_vbox)
        self.layout.addWidget(algo_gb)

        # Run
        self.run_btn = QPushButton("3. Train Model")
        self.run_btn.clicked.connect(self.run_pipeline)
        self.layout.addWidget(self.run_btn)

        # Output
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.data_status.setText(f"Loaded: {self.df.shape[0]} rows, {self.df.shape[1]} cols")
            self.settings.update("core_ml", "data_path", path)

    def generate_synthetic(self):
        task = self.task_cb.currentText()
        n_samples = self.samples_spin.value()
        self.df = SyntheticDataGenerator.tabular_data(task=task, n_samples=n_samples)
        self.data_status.setText(f"Synthetic {task} data: {self.df.shape[0]} samples")

    def run_pipeline(self):
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Load or generate data first.")
            return
        
        task = self.task_cb.currentText()
        algo = self.algo_cb.currentText()
        
        params = {
            "n_estimators": self.n_est.value(),
            "max_depth": self.max_depth.value(),
            "learning_rate": self.lr.value(),
            "random_state": 42
        }
        
        try:
            runner = PipelineRunner(task, algo, params)
            model, metrics = runner.run_tabular(self.df)
            
            self.output.clear()
            
            if "error" in metrics:
                self.output.setText(f"❌ Error: {metrics['error']}")
                return
            
            if task == "classification":
                self.output.append(f"✅ Model: {algo}")
                self.output.append(f"Accuracy: {metrics['accuracy']:.4f}")
                self.output.append(f"\nClassification Report:")
                self.output.append(metrics['report'])
            else:
                self.output.append(f"✅ Model: {algo}")
                self.output.append(f"MSE: {metrics['MSE']:.4f}")
            
            # Ayarları kaydet
            self.settings.update("core_ml", "algorithm", algo)
            self.settings.update("core_ml", "task", task)
            
        except Exception as e:
            self.output.setText(f"❌ Error: {str(e)}")