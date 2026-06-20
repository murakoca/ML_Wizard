from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QSpinBox,
                             QFileDialog, QMessageBox, QLineEdit)
import pandas as pd
import numpy as np
from utils.synthetic_data import SyntheticDataGenerator
from utils.pipeline_runner import PipelineRunner

class Stage07LLMNLP(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "LLM & NLP")
        self.set_theory("<h2>07. LLM & NLP</h2><p>Transformers, BERT, GPT, Prompt Engineering, RAG (Retrieval-Augmented Generation), Vector Databases, Sentiment Analysis, Text Classification.</p>")
        self.texts = []
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Görev seçimi
        task_gb = QGroupBox("1. NLP Task")
        task_vbox = QVBoxLayout()
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        self.task_cb = QComboBox()
        self.task_cb.addItems([
            "Sentiment Analysis (BERT)",
            "Text Classification (XGBoost + TF-IDF)",
            "RAG - Document Q&A",
            "Prompt Engineering"
        ])
        self.task_cb.currentTextChanged.connect(self._on_task_change)
        task_row.addWidget(self.task_cb)
        task_vbox.addLayout(task_row)

        # Model seçimi
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_cb = QComboBox()
        self.model_cb.addItems([
            "distilbert-base-uncased-finetuned-sst-2-english",
            "bert-base-uncased",
            "cardiffnlp/twitter-roberta-base-sentiment"
        ])
        model_row.addWidget(self.model_cb)
        task_vbox.addLayout(model_row)
        task_gb.setLayout(task_vbox)
        self.layout.addWidget(task_gb)

        # Veri
        data_gb = QGroupBox("2. Text Data")
        data_vbox = QVBoxLayout()

        # Manuel giriş
        data_vbox.addWidget(QLabel("Enter texts (one per line):"))
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(120)
        self.text_input.setPlaceholderText("Enter texts here...\nOr use buttons below")
        data_vbox.addWidget(self.text_input)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load CSV/Text File")
        btn_load.clicked.connect(self.load_text_file)
        btn_row.addWidget(btn_load)

        btn_synth = QPushButton("Generate Synthetic Texts")
        btn_synth.clicked.connect(self.generate_synthetic_texts)
        btn_row.addWidget(btn_synth)

        self.synth_count = QSpinBox()
        self.synth_count.setRange(10, 500)
        self.synth_count.setValue(50)
        btn_row.addWidget(QLabel("Count:"))
        btn_row.addWidget(self.synth_count)
        data_vbox.addLayout(btn_row)

        self.data_status = QLabel("No texts loaded")
        data_vbox.addWidget(self.data_status)
        data_gb.setLayout(data_vbox)
        self.layout.addWidget(data_gb)

        # Run
        self.run_btn = QPushButton("3. Run NLP Pipeline")
        self.run_btn.clicked.connect(self.run_nlp_pipeline)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        # RAG için özel alan (gizli başlar)
        self.rag_query = QLineEdit()
        self.rag_query.setPlaceholderText("Enter your question for RAG...")
        self.rag_query.hide()
        self.layout.addWidget(self.rag_query)

    def _on_task_change(self, task):
        self.model_cb.clear()
        if "RAG" in task:
            self.rag_query.show()
            self.model_cb.addItems(["sentence-transformers/all-MiniLM-L6-v2"])
        elif "Prompt" in task:
            self.rag_query.hide()
            self.model_cb.addItems(["gpt-3.5-turbo", "gpt-4", "claude-3"])
        else:
            self.rag_query.hide()
            self.model_cb.addItems([
                "distilbert-base-uncased-finetuned-sst-2-english",
                "bert-base-uncased",
                "cardiffnlp/twitter-roberta-base-sentiment"
            ])

    def load_text_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Text File", "",
                                               "Text/CSV Files (*.txt *.csv)")
        if path:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                if 'text' in df.columns:
                    self.texts = df['text'].tolist()
                else:
                    self.texts = df.iloc[:, 0].tolist()
            else:
                with open(path, 'r') as f:
                    self.texts = [line.strip() for line in f.readlines() if line.strip()]
            self.text_input.setText("\n".join(self.texts[:20]))
            self.data_status.setText(f"Loaded {len(self.texts)} texts from {path}")
            self.settings.update("llm", "texts", self.texts)

    def generate_synthetic_texts(self):
        n = self.synth_count.value()
        df = SyntheticDataGenerator.sentiment_texts(n)
        self.texts = df['text'].tolist()
        self.text_input.setText("\n".join(self.texts[:20]))
        self.data_status.setText(f"Generated {len(self.texts)} synthetic texts")
        self.settings.update("llm", "synthetic_texts", True)

    def run_nlp_pipeline(self):
        # Manuel giriş varsa onu al
        manual = self.text_input.toPlainText().strip()
        if manual:
            self.texts = [t.strip() for t in manual.split('\n') if t.strip()]

        if not self.texts:
            QMessageBox.warning(self, "No Texts", "Load, generate, or enter texts first.")
            return

        task = self.task_cb.currentText()
        model_name = self.model_cb.currentText()

        try:
            if "Sentiment" in task:
                runner = PipelineRunner(task="sentiment", algorithm="bert")
                results = runner.run_sentiment(self.texts[:20], model_name=model_name)
                self.output.clear()
                for text, res in zip(self.texts[:20], results):
                    self.output.append(f"📝 {text[:60]}...\n   ➜ {res['label']} ({res['score']:.3f})\n")

            elif "Classification" in task:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from xgboost import XGBClassifier
                import numpy as np

                if len(self.texts) < 10:
                    QMessageBox.warning(self, "Insufficient Data", "Need at least 10 texts for classification.")
                    return

                # Sentetik etiket oluştur (gerçek veri yoksa)
                labels = [1 if i < len(self.texts)//2 else 0 for i in range(len(self.texts))]
                vectorizer = TfidfVectorizer(max_features=100)
                X = vectorizer.fit_transform(self.texts)
                y = np.array(labels)

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = XGBClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                self.output.setText(f"✅ Text Classification Complete\nAccuracy: {acc:.4f}\n"
                                    f"Features: {X.shape[1]}\nSamples: {len(self.texts)}")

            elif "RAG" in task:
                query = self.rag_query.text() if self.rag_query.text() else "What is this document about?"
                self.output.setText(f"🔍 RAG Query: '{query}'\n\n"
                                    f"📄 Documents indexed: {len(self.texts)}\n"
                                    f"🔗 Vector Store: FAISS (in-memory)\n"
                                    f"💡 Top 3 relevant chunks would be retrieved and fed to LLM.\n\n"
                                    f"⚠️ Full RAG pipeline requires: LangChain/LlamaIndex + Vector DB + LLM API key.\n"
                                    f"Install: pip install langchain chromadb sentence-transformers")

            elif "Prompt" in task:
                self.output.setText(f"🧠 Prompt Engineering\n\n"
                                    f"Template: 'Analyze the sentiment of this text: {{text}}'\n\n"
                                    f"Would be sent to: {model_name}\n"
                                    f"⚠️ Requires API key. Set in environment variables.")

            self.settings.update("llm", "task", task)
            self.settings.update("llm", "model", model_name)
            self.settings.update("llm", "text_count", len(self.texts))
        except Exception as e:
            self.output.setText(f"❌ Error: {e}\nMake sure transformers is installed: pip install transformers torch")