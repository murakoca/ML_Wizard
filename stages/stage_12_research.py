from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QCheckBox,
                             QLineEdit, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt
import json

class Stage12Research(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Research Literacy & Continuous Learning")
        self.set_theory("<h2>12. Research Literacy & Continuous Learning</h2><p>Paper Reading, Experiment Design, Reproducibility, Benchmarks, Ablation Studies, Papers With Code, arXiv, Weights & Biases.</p>")
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Araştırma kaynakları
        sources_gb = QGroupBox("1. Research Sources")
        sources_vbox = QVBoxLayout()
        self.source_checks = {}
        sources = [
            "arXiv (cs.AI, cs.CV, cs.CL, cs.MA)",
            "Papers With Code",
            "OpenReview",
            "Hugging Face Daily Papers",
            "Google Scholar Alerts",
            "NeurIPS / ICML / ICLR Proceedings",
            "GitHub Trending AI Repos"
        ]
        for src in sources:
            cb = QCheckBox(src)
            self.source_checks[src] = cb
            sources_vbox.addWidget(cb)
        sources_gb.setLayout(sources_vbox)
        self.layout.addWidget(sources_gb)

        # Deney tasarımı
        experiment_gb = QGroupBox("2. Experiment Design")
        exp_vbox = QVBoxLayout()
        exp_vbox.addWidget(QLabel("Experiment Name:"))
        self.exp_name = QLineEdit("my_experiment")
        exp_vbox.addWidget(self.exp_name)

        exp_vbox.addWidget(QLabel("Hypothesis:"))
        self.hypothesis = QTextEdit()
        self.hypothesis.setMaximumHeight(60)
        self.hypothesis.setPlaceholderText("E.g., Adding attention mechanism improves accuracy by 5%...")
        exp_vbox.addWidget(self.hypothesis)

        exp_vbox.addWidget(QLabel("Metrics to track:"))
        self.metrics_input = QLineEdit("accuracy, loss, F1-score, inference_time")
        exp_vbox.addWidget(self.metrics_input)

        self.ablation_check = QCheckBox("Include Ablation Study")
        exp_vbox.addWidget(self.ablation_check)
        self.repro_check = QCheckBox("Ensure Reproducibility (seed, env, config)")
        self.repro_check.setChecked(True)
        exp_vbox.addWidget(self.repro_check)
        experiment_gb.setLayout(exp_vbox)
        self.layout.addWidget(experiment_gb)

        # Benchmark
        benchmark_gb = QGroupBox("3. Benchmarks")
        bench_vbox = QVBoxLayout()
        self.bench_cb = QComboBox()
        self.bench_cb.addItems([
            "ImageNet (CV)",
            "COCO (Object Detection)",
            "GLUE/SuperGLUE (NLP)",
            "MMLU (LLM Knowledge)",
            "HumanEval (Code Generation)",
            "Custom Benchmark"
        ])
        bench_vbox.addWidget(QLabel("Select Benchmark:"))
        bench_vbox.addWidget(self.bench_cb)
        benchmark_gb.setLayout(bench_vbox)
        self.layout.addWidget(benchmark_gb)

        # Run
        self.run_btn = QPushButton("4. Generate Research Plan")
        self.run_btn.clicked.connect(self.generate_plan)
        self.layout.addWidget(self.run_btn)

        # Final Özet
        self.summary_btn = QPushButton("📋 Generate Complete Roadmap Summary")
        self.summary_btn.clicked.connect(self.generate_summary)
        self.layout.addWidget(self.summary_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def generate_plan(self):
        selected_sources = [s for s, cb in self.source_checks.items() if cb.isChecked()]
        exp_name = self.exp_name.text()
        hypothesis = self.hypothesis.toPlainText().strip()
        metrics = self.metrics_input.text()
        benchmark = self.bench_cb.currentText()

        self.output.clear()
        self.output.append(f"🔬 Research Plan: {exp_name}\n{'='*60}")

        self.output.append(f"\n📚 Literature Review Sources:")
        for src in selected_sources:
            self.output.append(f"  ✅ {src}")

        self.output.append(f"\n🧪 Hypothesis:")
        self.output.append(f"  {hypothesis if hypothesis else 'Not specified'}")

        self.output.append(f"\n📊 Metrics:")
        self.output.append(f"  {metrics}")

        self.output.append(f"\n🏆 Benchmark:")
        self.output.append(f"  {benchmark}")

        if self.ablation_check.isChecked():
            self.output.append(f"\n🔪 Ablation Study Plan:")
            self.output.append(f"  1. Baseline model")
            self.output.append(f"  2. Remove component A → measure impact")
            self.output.append(f"  3. Remove component B → measure impact")
            self.output.append(f"  4. Ablation table + discussion")

        if self.repro_check.isChecked():
            self.output.append(f"\n♻️ Reproducibility Checklist:")
            self.output.append(f"  ✅ Fixed random seeds (42)")
            self.output.append(f"  ✅ Environment: requirements.txt with versions")
            self.output.append(f"  ✅ Config files in YAML/JSON")
            self.output.append(f"  ✅ Weights & Biases for experiment tracking")
            self.output.append(f"  ✅ Code on GitHub with README")

        self.output.append(f"\n📝 Paper Structure:")
        self.output.append(f"  1. Abstract")
        self.output.append(f"  2. Introduction & Related Work")
        self.output.append(f"  3. Methodology")
        self.output.append(f"  4. Experiments & Results")
        self.output.append(f"  5. Ablation Study")
        self.output.append(f"  6. Conclusion")

        self.output.append(f"\n🔗 Useful Tools:")
        self.output.append(f"  - Weights & Biases: wandb.ai")
        self.output.append(f"  - Papers With Code: paperswithcode.com")
        self.output.append(f"  - arXiv Sanity: arxiv-sanity-lite.com")
        self.output.append(f"  - Connected Papers: connectedpapers.com")

        self.settings.update("research", "experiment", exp_name)
        self.settings.update("research", "benchmark", benchmark)

    def generate_summary(self):
        """Tüm yol haritasının özetini çıkarır"""
        self.output.clear()
        self.output.append(f"🎓 AI ENGINEER ROADMAP 2026 - COMPLETE SUMMARY\n{'='*60}")

        stages_summary = [
            ("01", "Mathematics", "Linear Algebra, Calculus, Probability"),
            ("02", "Python & SE", "Pythonic Design, Git, Docker, Testing"),
            ("03", "Data Engineering", "ETL, Feature Engineering, Validation"),
            ("04", "Classical ML", "Regression, Classification, Ensembles"),
            ("05", "Core ML", "XGBoost, LightGBM, Hyperparameter Tuning"),
            ("06", "DL & CV", "CNN, YOLO, SAM, OCR, Transformers"),
            ("07", "LLM & NLP", "BERT, GPT, RAG, Sentiment Analysis"),
            ("08", "Agentic AI", "Tool Calling, Multi-Agent, MCP, A2A"),
            ("09", "RL", "Q-Learning, PPO, RLHF, GRPO"),
            ("10", "Interpretability", "SHAP, LIME, Fairness, Guardrails"),
            ("11", "MLOps/LLMOps", "Docker, K8s, CI/CD, Monitoring"),
            ("12", "Research", "Paper Reading, Experiments, Benchmarks")
        ]

        for num, name, topics in stages_summary:
            self.output.append(f"\n✅ Stage {num}: {name}")
            self.output.append(f"   {topics}")

        self.output.append(f"\n{'='*60}")
        self.output.append(f"\n🚀 FINAL GIT COMMANDS:")
        self.output.append(f"cd ai_roadmap_app")
        self.output.append(f"git init")
        self.output.append(f"git add .")
        self.output.append(f"git commit -m \"Complete AI Engineer Roadmap 2026\"")
        self.output.append(f"git tag v1.0.0")
        self.output.append(f"git push origin main --tags")
        self.output.append(f"\n📦 Create requirements.txt:")
        self.output.append(f"pip freeze > requirements.txt")
        self.output.append(f"git add requirements.txt")
        self.output.append(f"git commit -m \"Added frozen requirements\"")

        self.output.append(f"\n{'='*60}")
        self.output.append(f"🎉 Congratulations! You've completed the full roadmap.")
        self.output.append(f"Next steps: Build a portfolio project, contribute to open source,")
        self.output.append(f"write a paper, deploy a model to production.")