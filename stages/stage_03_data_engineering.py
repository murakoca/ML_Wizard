from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QFileDialog, QComboBox, QSpinBox)
import pandas as pd
import os
from utils.synthetic_data import SyntheticDataGenerator

class Stage03DataEngineering(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Data Engineering & Feature Engineering")
        self.set_theory("<h2>03. Data Engineering</h2><p>ETL Pipelines, Sampling, Data Validation, Feature Stores, Time-Series, Missing Data Strategies.</p>")
        self.df = None
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Veri yükleme
        load_gb = QGroupBox("Load Data")
        load_vbox = QVBoxLayout()
        btn_csv = QPushButton("Load CSV File")
        btn_csv.clicked.connect(self.load_csv)
        load_vbox.addWidget(btn_csv)

        self.data_info = QLabel("No data loaded yet.")
        load_vbox.addWidget(self.data_info)
        load_gb.setLayout(load_vbox)
        self.layout.addWidget(load_gb)

        # Sentetik veri
        synth_gb = QGroupBox("Generate Synthetic Data")
        synth_vbox = QVBoxLayout()
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Samples:"))
        self.n_samples = QSpinBox()
        self.n_samples.setRange(100, 100000)
        self.n_samples.setValue(1000)
        row_layout.addWidget(self.n_samples)
        row_layout.addWidget(QLabel("Features:"))
        self.n_features = QSpinBox()
        self.n_features.setRange(5, 200)
        self.n_features.setValue(20)
        row_layout.addWidget(self.n_features)
        synth_vbox.addLayout(row_layout)

        self.task_cb = QComboBox()
        self.task_cb.addItems(["classification", "regression"])
        synth_vbox.addWidget(QLabel("Task:"))
        synth_vbox.addWidget(self.task_cb)

        btn_synth = QPushButton("Generate Synthetic Tabular Data")
        btn_synth.clicked.connect(self.generate_synthetic)
        synth_vbox.addWidget(btn_synth)
        synth_gb.setLayout(synth_vbox)
        self.layout.addWidget(synth_gb)

        # Önizleme
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(200)
        self.layout.addWidget(self.preview)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.df = pd.read_csv(path)
                self.data_info.setText(f"✅ Loaded: {path}\nShape: {self.df.shape}")
                self.preview.setText(self.df.head().to_string())
                self.settings.update("data_eng", "data_path", path)
            except Exception as e:
                self.data_info.setText(f"❌ Error: {e}")

    def generate_synthetic(self):
        n_samples = self.n_samples.value()
        n_features = self.n_features.value()
        task = self.task_cb.currentText()
        self.df = SyntheticDataGenerator.tabular_data(
            task=task, n_samples=n_samples, n_features=n_features
        )
        self.data_info.setText(f"✅ Synthetic data generated. Shape: {self.df.shape}")
        self.preview.setText(self.df.head().to_string())
        self.settings.update("data_eng", "synthetic", True)
        self.settings.update("data_eng", "task", task)