from stages.stage_base import StageWidget
from PyQt5.QtWidgets import QLabel, QCheckBox, QGroupBox, QVBoxLayout
from PyQt5.QtCore import Qt

class Stage01Math(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Mathematics for ML")
        self.set_theory("<h2>01. Mathematics for ML</h2><p>Linear Algebra, Multivariable Calculus, Matrix Calculus, Basic Graph Theory, Game Theory.</p>")
        self._build_ui()
        self.add_navigation()

    def _build_ui(self):
        gb = QGroupBox("Select topics you know:")
        vbox = QVBoxLayout()
        self.checks = {}
        for t in ["Linear Algebra", "Multivariable Calculus", "Matrix Calculus", "Graph Theory", "Game Theory"]:
            cb = QCheckBox(t)
            self.checks[t] = cb
            vbox.addWidget(cb)
        gb.setLayout(vbox)
        self.layout.addWidget(gb)

    def save_and_next(self):
        for t, cb in self.checks.items():
            self.settings.update("math", t, cb.isChecked())
        self.next_stage.emit()