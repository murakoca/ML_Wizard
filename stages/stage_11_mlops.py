from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QCheckBox,
                             QLineEdit, QMessageBox)
import subprocess
import sys
import os

class Stage11MLOps(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "MLOps & LLMOps")
        self.set_theory("<h2>11. MLOps & Deployment</h2><p>Containerization, CI/CD, Model Versioning, Model Serving (REST/gRPC), Monitoring, Drift Detection, Rollback Strategies, LLMOps, AgentOps.</p>")
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Deployment stratejisi
        deploy_gb = QGroupBox("1. Deployment Strategy")
        deploy_vbox = QVBoxLayout()
        deploy_row = QHBoxLayout()
        deploy_row.addWidget(QLabel("Strategy:"))
        self.deploy_cb = QComboBox()
        self.deploy_cb.addItems([
            "Docker Container",
            "FastAPI REST Endpoint",
            "Kubernetes Deployment",
            "Serverless (AWS Lambda)",
            "CI/CD Pipeline",
            "Model Registry (MLflow)"
        ])
        deploy_row.addWidget(self.deploy_cb)
        deploy_vbox.addLayout(deploy_row)
        deploy_gb.setLayout(deploy_vbox)
        self.layout.addWidget(deploy_gb)

        # Model yönetimi
        model_gb = QGroupBox("2. Model Management")
        model_vbox = QVBoxLayout()
        self.mlflow_check = QCheckBox("Use MLflow for model tracking")
        model_vbox.addWidget(self.mlflow_check)
        self.version_check = QCheckBox("Enable model versioning (DVC)")
        model_vbox.addWidget(self.version_check)
        model_vbox.addWidget(QLabel("Model Name:"))
        self.model_name = QLineEdit("my_model")
        model_vbox.addWidget(self.model_name)
        model_vbox.addWidget(QLabel("Version:"))
        self.model_version = QLineEdit("1.0.0")
        model_vbox.addWidget(self.model_version)
        model_gb.setLayout(model_vbox)
        self.layout.addWidget(model_gb)

        # Monitoring
        monitor_gb = QGroupBox("3. Monitoring & Observability")
        monitor_vbox = QVBoxLayout()
        self.monitor_checks = {}
        for tool in ["Prometheus Metrics", "Grafana Dashboard", "Drift Detection",
                     "Latency Monitoring", "Token Usage (LLM)", "Agent Trace (AgentOps)"]:
            cb = QCheckBox(tool)
            self.monitor_checks[tool] = cb
            monitor_vbox.addWidget(cb)
        monitor_gb.setLayout(monitor_vbox)
        self.layout.addWidget(monitor_gb)

        # Run
        self.run_btn = QPushButton("4. Generate Deployment Config")
        self.run_btn.clicked.connect(self.generate_deployment)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

        # Docker kontrol
        self.docker_check_btn = QPushButton("Check Docker Installation")
        self.docker_check_btn.clicked.connect(self.check_docker)
        self.layout.addWidget(self.docker_check_btn)

    def check_docker(self):
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                QMessageBox.information(self, "Docker", f"Docker detected:\n{result.stdout}")
            else:
                QMessageBox.warning(self, "Docker", "Docker not found or not running.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Docker", "Docker not installed.")
        except Exception as e:
            QMessageBox.warning(self, "Docker", f"Error: {e}")

    def generate_deployment(self):
        strategy = self.deploy_cb.currentText()
        model_name = self.model_name.text()
        version = self.model_version.text()
        selected_monitoring = [m for m, cb in self.monitor_checks.items() if cb.isChecked()]

        self.output.clear()
        self.output.append(f"🚀 Deployment Configuration\n{'='*50}")
        self.output.append(f"Strategy: {strategy}")
        self.output.append(f"Model: {model_name} v{version}")
        self.output.append(f"\n{'='*50}")

        if "Docker" in strategy:
            dockerfile = f"""
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            self.output.append(f"\n📦 Dockerfile:\n{dockerfile}")

            docker_compose = f"""
# docker-compose.yml
version: '3.8'
services:
  {model_name}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_VERSION={version}
    volumes:
      - ./models:/app/models
"""
            self.output.append(f"\n📦 docker-compose.yml:\n{docker_compose}")

        if "FastAPI" in strategy:
            fastapi_code = f"""
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="{model_name}")
model = joblib.load("models/{model_name}_{version}.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    version: str = "{version}"

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    pred = model.predict([request.features])
    return PredictionResponse(prediction=float(pred[0]))
"""
            self.output.append(f"\n🐍 FastAPI Server:\n{fastapi_code}")

        if "Kubernetes" in strategy:
            k8s_manifest = f"""
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {model_name}
  template:
    metadata:
      labels:
        app: {model_name}
    spec:
      containers:
      - name: {model_name}
        image: {model_name}:{version}
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-service
spec:
  selector:
    app: {model_name}
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
            self.output.append(f"\n☸️ Kubernetes Manifest:\n{k8s_manifest}")

        if "MLflow" in self.mlflow_check.text() or self.mlflow_check.isChecked():
            self.output.append(f"\n📊 MLflow Tracking:")
            self.output.append(f"  mlflow server --host 0.0.0.0 --port 5000")
            self.output.append(f"  MLflow UI: http://localhost:5000")
            self.output.append(f"  Log params, metrics, artifacts")

        # Monitoring
        if selected_monitoring:
            self.output.append(f"\n📈 Monitoring Setup:")
            for tool in selected_monitoring:
                if "Prometheus" in tool:
                    self.output.append(f"  - Prometheus: Scrape metrics from /metrics endpoint")
                elif "Grafana" in tool:
                    self.output.append(f"  - Grafana: Dashboard JSON template provided")
                elif "Drift" in tool:
                    self.output.append(f"  - Evidently AI for drift detection")
                    self.output.append(f"    pip install evidently")
                elif "Token" in tool:
                    self.output.append(f"  - LLM Token tracking with LangFuse")
                    self.output.append(f"    pip install langfuse")
                elif "Agent" in tool:
                    self.output.append(f"  - AgentOps tracing for agent decision chains")
                    self.output.append(f"    pip install agentops")

        # CI/CD
        if "CI/CD" in strategy:
            github_actions = f"""
# .github/workflows/deploy.yml
name: Deploy {model_name}
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t {model_name}:{version} .
      - name: Push to registry
        run: docker push registry.example.com/{model_name}:{version}
      - name: Deploy to Kubernetes
        run: kubectl apply -f deployment.yaml
"""
            self.output.append(f"\n🔄 GitHub Actions CI/CD:\n{github_actions}")

        self.output.append(f"\n{'='*50}")
        self.output.append(f"\n✅ Git Commands for this stage:")
        self.output.append(f"git add .")
        self.output.append(f"git commit -m \"Added MLOps configuration for {model_name} v{version}\"")
        self.output.append(f"git tag v{version}")
        self.output.append(f"git push origin main --tags")

        self.settings.update("mlops", "strategy", strategy)
        self.settings.update("mlops", "model_name", model_name)
        self.settings.update("mlops", "version", version)