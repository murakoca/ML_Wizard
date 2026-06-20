from PyQt5.QtWidgets import (QMainWindow, QListWidget, QStackedWidget,
                             QHBoxLayout, QWidget, QListWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt
from functools import partial
from utils.settings import Settings

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle("AI Engineer Roadmap 2026 - Interactive Guide")
        self.resize(1300, 850)
        self._init_ui()
        self._connect_list()

    def _init_ui(self):
        # Lazy import stage'leri
        from stages.stage_01_math import Stage01Math
        from stages.stage_02_python import Stage02Python
        from stages.stage_03_data_engineering import Stage03DataEngineering
        from stages.stage_04_classical_ml import Stage04ClassicalML
        from stages.stage_05_core_ml import Stage05CoreML
        from stages.stage_06_deep_learning_cv import Stage06DeepLearningCV
        from stages.stage_07_llm_nlp import Stage07LLMNLP
        from stages.stage_08_agentic_ai import Stage08AgenticAI
        from stages.stage_09_reinforcement_learning import Stage09RL
        from stages.stage_10_interpretability import Stage10Interpretability
        from stages.stage_11_mlops import Stage11MLOps
        from stages.stage_12_research import Stage12Research
        
        central = QWidget()
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)
        hbox.setContentsMargins(0, 0, 0, 0)

        # Sidebar - aşama listesi
        self.stage_list = QListWidget()
        self.stage_list.setMaximumWidth(260)
        self.stage_names = [
            "01. Mathematics for ML",
            "02. Python & Software Eng.",
            "03. Data Engineering",
            "04. Classical ML",
            "05. Core ML (Algorithms)",
            "06. Deep Learning & CV",
            "07. LLM & NLP",
            "08. Agentic AI",
            "09. Reinforcement Learning",
            "10. Model Interpretability",
            "11. MLOps & Deployment",
            "12. Research Literacy"
        ]
        for i, name in enumerate(self.stage_names):
            item = QListWidgetItem(f"  {name}")
            item.setData(Qt.UserRole, i)
            self.stage_list.addItem(item)
        hbox.addWidget(self.stage_list)

        # Stacked widget - tüm 12 stage
        self.stack = QStackedWidget()
        self.stage_widgets = [
            Stage01Math(self.settings),
            Stage02Python(self.settings),
            Stage03DataEngineering(self.settings),
            Stage04ClassicalML(self.settings),
            Stage05CoreML(self.settings),
            Stage06DeepLearningCV(self.settings),
            Stage07LLMNLP(self.settings),
            Stage08AgenticAI(self.settings),
            Stage09RL(self.settings),
            Stage10Interpretability(self.settings),
            Stage11MLOps(self.settings),
            Stage12Research(self.settings)
        ]

        for i, w in enumerate(self.stage_widgets):
            if hasattr(w, 'next_stage'):
                w.next_stage.connect(partial(self._handle_next, i))
            if hasattr(w, 'prev_stage'):
                w.prev_stage.connect(partial(self._handle_prev, i))
            self.stack.addWidget(w)

        hbox.addWidget(self.stack)

    def _connect_list(self):
        """Sidebar liste tıklamasını stack widget'a bağla"""
        self.stage_list.currentRowChanged.connect(self._safe_switch)

    def _safe_switch(self, idx):
        """Güvenli aşama geçişi"""
        if idx >= 0 and idx < self.stack.count():
            self.stack.setCurrentIndex(idx)

    def _handle_next(self, current_idx):
        """Sonraki aşamaya geç"""
        next_idx = current_idx + 1
        if next_idx < self.stack.count():
            self.stack.setCurrentIndex(next_idx)
            self.stage_list.setCurrentRow(next_idx)
            QMessageBox.information(self, "✅ Stage Completed",
                f"<b>Stage {current_idx + 1} completed!</b>\n\n"
                f"Run the following Git commands in your terminal:\n\n"
                f"<code style='color:#4CAF50;'>git init</code><br>"
                f"<code style='color:#4CAF50;'>git add .</code><br>"
                f"<code style='color:#4CAF50;'>git commit -m \"Completed stage {current_idx + 1}: "
                f"{self.stage_names[current_idx].split('. ')[1]}\"</code>",
                QMessageBox.Ok)

    def _handle_prev(self, current_idx):
        """Önceki aşamaya dön"""
        prev_idx = current_idx - 1
        if prev_idx >= 0:
            self.stack.setCurrentIndex(prev_idx)
            self.stage_list.setCurrentRow(prev_idx)

    def keyPressEvent(self, event):
        """Klavye kısayolları: Sağ ok = ileri, Sol ok = geri"""
        if event.key() == Qt.Key_Right:
            current = self.stack.currentIndex()
            self._handle_next(current)
        elif event.key() == Qt.Key_Left:
            current = self.stack.currentIndex()
            self._handle_prev(current)