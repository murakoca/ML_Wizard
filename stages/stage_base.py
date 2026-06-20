from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import pyqtSignal

class StageWidget(QWidget):
    next_stage = pyqtSignal()
    prev_stage = pyqtSignal()

    def __init__(self, settings, stage_name):
        super().__init__()
        self.settings = settings
        self.stage_name = stage_name
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self._add_theory()

    def _add_theory(self):
        self.theory_browser = QTextBrowser()
        self.theory_browser.setMaximumHeight(150)
        self.layout.addWidget(self.theory_browser)

    def set_theory(self, html):
        self.theory_browser.setHtml(html)

    def add_navigation(self, next_enabled=True):
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.prev_stage.emit)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(next_enabled)
        self.next_btn.clicked.connect(self.next_stage.emit)
        nav_layout.addWidget(self.next_btn)
        self.layout.addLayout(nav_layout)

    def show_git_info(self, stage_num):
        msg = QMessageBox()
        msg.setWindowTitle("Git Commit")
        msg.setText(f"Stage {stage_num} completed!\nRun the following commands:")
        msg.setDetailedText(f"git init\ngit add .\ngit commit -m \"Completed stage {stage_num}: {self.stage_name}\"")
        msg.exec_()