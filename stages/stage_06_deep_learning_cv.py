from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QSpinBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt
import os
import numpy as np
from PIL import Image
from utils.synthetic_data import SyntheticDataGenerator
from utils.pipeline_runner import PipelineRunner

class Stage06DeepLearningCV(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Deep Learning & Computer Vision")
        self.set_theory("<h2>06. Deep Learning & Computer Vision</h2><p>CNN, RNN, LSTM, Transformers, YOLO, SAM, OCR (TrOCR), Object Detection, Segmentation, Object Tracking.</p>")
        self.image_paths = []
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Görev seçimi
        task_gb = QGroupBox("1. Vision Task")
        task_vbox = QVBoxLayout()
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        self.task_cb = QComboBox()
        self.task_cb.addItems(["Object Detection (YOLO)", "Segmentation (SAM)", "OCR (TrOCR)"])
        self.task_cb.currentTextChanged.connect(self._on_task_change)
        task_row.addWidget(self.task_cb)
        task_vbox.addLayout(task_row)

        # Model seçimi
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_cb = QComboBox()
        self.model_cb.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        model_row.addWidget(self.model_cb)
        task_vbox.addLayout(model_row)

        # Parametreler
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        param_row.addWidget(self.conf_spin)
        param_row.addWidget(QLabel("IoU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        param_row.addWidget(self.iou_spin)
        task_vbox.addLayout(param_row)
        task_gb.setLayout(task_vbox)
        self.layout.addWidget(task_gb)

        # Görüntü yükleme
        img_gb = QGroupBox("2. Images")
        img_vbox = QVBoxLayout()
        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load Images (jpg/png/tiff)")
        btn_load.clicked.connect(self.load_images)
        btn_row.addWidget(btn_load)
        btn_synth = QPushButton("Generate Synthetic Images")
        btn_synth.clicked.connect(self.generate_synthetic_images)
        btn_row.addWidget(btn_synth)
        img_vbox.addLayout(btn_row)

        self.img_list = QListWidget()
        img_vbox.addWidget(self.img_list)

        self.img_count = QLabel("0 images loaded")
        img_vbox.addWidget(self.img_count)
        img_gb.setLayout(img_vbox)
        self.layout.addWidget(img_gb)

        # Run butonu
        self.run_btn = QPushButton("3. Run Vision Pipeline")
        self.run_btn.clicked.connect(self.run_vision_pipeline)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def _on_task_change(self, task):
        self.model_cb.clear()
        if "YOLO" in task:
            self.model_cb.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        elif "SAM" in task:
            self.model_cb.addItems(["sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth"])
        elif "OCR" in task:
            self.model_cb.addItems(["microsoft/trocr-base-printed", "microsoft/trocr-large-printed"])

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.jpg *.jpeg *.png *.tiff *.tif *.bmp)"
        )
        if paths:
            self.image_paths = paths
            self.img_list.clear()
            for p in paths:
                self.img_list.addItem(os.path.basename(p))
            self.img_count.setText(f"{len(self.image_paths)} images loaded")
            self.settings.update("cv", "image_paths", self.image_paths)

    def generate_synthetic_images(self):
        n = 5
        paths = SyntheticDataGenerator.create_random_images(n=n, output_dir="synthetic_images")
        self.image_paths = paths
        self.img_list.clear()
        for p in paths:
            self.img_list.addItem(os.path.basename(p))
        self.img_count.setText(f"{n} synthetic images generated")
        self.settings.update("cv", "synthetic_images", True)

    def run_vision_pipeline(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Load or generate images first.")
            return
        task = self.task_cb.currentText()
        model_name = self.model_cb.currentText()
        try:
            if "YOLO" in task:
                runner = PipelineRunner(task="detection", algorithm="yolo")
                results = runner.run_yolo(self.image_paths, model_name=model_name)
                total_objects = 0
                for r in results:
                    boxes = r[0].boxes
                    if boxes is not None:
                        total_objects += len(boxes)
                self.output.setText(f"✅ YOLO Detection Complete\n"
                                    f"Images processed: {len(self.image_paths)}\n"
                                    f"Total objects detected: {total_objects}\n"
                                    f"Results saved in runs/detect/")
            elif "SAM" in task:
                self.output.setText("⚠️ SAM segmentation requires GPU. Image masks would be generated here.")
            elif "OCR" in task:
                self.output.setText("⚠️ OCR with TrOCR: Text extraction would be performed here.\n"
                                    "Sample output: 'Detected text from images...'")
            self.settings.update("cv", "task", task)
            self.settings.update("cv", "model", model_name)
        except Exception as e:
            self.output.setText(f"❌ Error: {e}\nMake sure ultralytics is installed: pip install ultralytics")