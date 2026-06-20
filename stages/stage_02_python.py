from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QCheckBox, QGroupBox, QVBoxLayout,
                             QTextEdit, QPushButton, QHBoxLayout, QFileDialog)
import subprocess
import sys

class Stage02Python(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Python & Software Engineering")
        self.set_theory("<h2>02. Programming & Software Engineering</h2><p>Pythonic Design, Data Structures, Testing, Design Patterns, Git, Virtual Environments, Packaging.</p>")
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Bilgi seçimi
        gb = QGroupBox("Which topics are you familiar with?")
        vbox = QVBoxLayout()
        self.checks = {}
        topics = ["Pythonic Design", "Data Structures & Algorithms", "Testing (pytest/unittest)",
                  "Design Patterns", "Git", "Virtual Environments", "Package Management (Poetry/pip)"]
        for t in topics:
            cb = QCheckBox(t)
            self.checks[t] = cb
            vbox.addWidget(cb)
        gb.setLayout(vbox)
        self.layout.addWidget(gb)

        # Python versiyon kontrolü
        self.py_label = QLabel()
        self.check_python()
        self.layout.addWidget(self.py_label)

        # Pip listesi
        self.pip_btn = QPushButton("Show Installed Packages (pip list)")
        self.pip_btn.clicked.connect(self.show_pip_list)
        self.layout.addWidget(self.pip_btn)

        self.pip_output = QTextEdit()
        self.pip_output.setReadOnly(True)
        self.pip_output.setMaximumHeight(120)
        self.layout.addWidget(self.pip_output)

        # Requirements.txt yükleme
        self.req_btn = QPushButton("Load requirements.txt")
        self.req_btn.clicked.connect(self.load_requirements)
        self.layout.addWidget(self.req_btn)

        self.req_output = QTextEdit()
        self.req_output.setReadOnly(True)
        self.req_output.setMaximumHeight(100)
        self.layout.addWidget(self.req_output)

    def check_python(self):
        ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.py_label.setText(f"✅ Python {ver} detected")

    def show_pip_list(self):
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                                    capture_output=True, text=True, timeout=30)
            self.pip_output.setText(result.stdout)
        except Exception as e:
            self.pip_output.setText(f"Error: {e}")

    def load_requirements(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open requirements.txt", "", "Text Files (*.txt)")
        if path:
            with open(path, 'r') as f:
                content = f.read()
            self.req_output.setText(content)
            self.settings.update("python", "requirements_file", path)