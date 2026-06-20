from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QCheckBox,
                             QFileDialog, QMessageBox, QLineEdit, QListWidget)
from PyQt5.QtCore import Qt
import json

class Stage08AgenticAI(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Agentic AI & Multi-Agent Systems")
        self.set_theory("<h2>08. Agentic AI</h2><p>Tool Calling, Function Calling, ReAct, Plan-and-Execute, Multi-Agent Systems (MAS), MCP (Model Context Protocol), A2A (Agent-to-Agent).</p>")
        self.agent_config = {}
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Ajan tipi
        agent_gb = QGroupBox("1. Agent Architecture")
        agent_vbox = QVBoxLayout()

        arch_row = QHBoxLayout()
        arch_row.addWidget(QLabel("Architecture:"))
        self.arch_cb = QComboBox()
        self.arch_cb.addItems([
            "Single Agent (Tool Calling)",
            "ReAct Agent",
            "Multi-Agent System (CrewAI style)",
            "MCP Server Agent",
            "A2A Protocol Agent"
        ])
        self.arch_cb.currentTextChanged.connect(self._on_arch_change)
        arch_row.addWidget(self.arch_cb)
        agent_vbox.addLayout(arch_row)

        # LLM seçimi
        llm_row = QHBoxLayout()
        llm_row.addWidget(QLabel("LLM Backend:"))
        self.llm_cb = QComboBox()
        self.llm_cb.addItems(["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama-3", "mistral"])
        llm_row.addWidget(self.llm_cb)
        agent_vbox.addLayout(llm_row)
        agent_gb.setLayout(agent_vbox)
        self.layout.addWidget(agent_gb)

        # Araçlar (Toolbox)
        tool_gb = QGroupBox("2. Available Tools")
        tool_vbox = QVBoxLayout()
        self.tool_checks = {}
        tools = ["Web Search", "Calculator", "Code Interpreter", "File Reader",
                 "Database Query", "API Caller", "Image Generator", "PDF Parser"]
        for tool in tools:
            cb = QCheckBox(tool)
            self.tool_checks[tool] = cb
            tool_vbox.addWidget(cb)
        tool_gb.setLayout(tool_vbox)
        self.layout.addWidget(tool_gb)

        # Ajan hedefi
        goal_gb = QGroupBox("3. Agent Goal")
        goal_vbox = QVBoxLayout()
        goal_vbox.addWidget(QLabel("What should the agent do?"))
        self.goal_input = QTextEdit()
        self.goal_input.setMaximumHeight(80)
        self.goal_input.setPlaceholderText("E.g.: Analyze this CSV, find patterns, and generate a report...")
        goal_vbox.addWidget(self.goal_input)
        goal_gb.setLayout(goal_vbox)
        self.layout.addWidget(goal_gb)

        # Çok ajanlı yapılandırma (gizli)
        self.multi_agent_gb = QGroupBox("4. Multi-Agent Configuration")
        multi_vbox = QVBoxLayout()
        multi_vbox.addWidget(QLabel("Agent Roles (one per line):"))
        self.roles_input = QTextEdit()
        self.roles_input.setMaximumHeight(60)
        self.roles_input.setPlaceholderText("Researcher\nAnalyst\nWriter")
        multi_vbox.addWidget(self.roles_input)
        multi_vbox.addWidget(QLabel("Task:"))
        self.mas_task = QTextEdit()
        self.mas_task.setMaximumHeight(60)
        self.mas_task.setPlaceholderText("Research topic X, analyze findings, write report...")
        multi_vbox.addWidget(self.mas_task)
        self.multi_agent_gb.setLayout(multi_vbox)
        self.multi_agent_gb.hide()
        self.layout.addWidget(self.multi_agent_gb)

        # Run
        self.run_btn = QPushButton("4. Execute Agent")
        self.run_btn.clicked.connect(self.run_agent)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def _on_arch_change(self, arch):
        if "Multi" in arch:
            self.multi_agent_gb.show()
        else:
            self.multi_agent_gb.hide()

    def run_agent(self):
        arch = self.arch_cb.currentText()
        llm = self.llm_cb.currentText()
        selected_tools = [t for t, cb in self.tool_checks.items() if cb.isChecked()]
        goal = self.goal_input.toPlainText().strip()

        if not goal:
            QMessageBox.warning(self, "No Goal", "Please enter an agent goal.")
            return

        self.agent_config = {
            "architecture": arch,
            "llm": llm,
            "tools": selected_tools,
            "goal": goal
        }

        self.output.clear()
        self.output.append(f"🤖 Agent Configuration\n{'='*50}")
        self.output.append(f"Architecture: {arch}")
        self.output.append(f"LLM Backend: {llm}")
        self.output.append(f"Tools: {', '.join(selected_tools) if selected_tools else 'None'}")
        self.output.append(f"Goal: {goal}")
        self.output.append(f"\n{'='*50}")

        if "Multi" in arch:
            roles = self.roles_input.toPlainText().strip().split('\n')
            mas_task = self.mas_task.toPlainText().strip()
            self.output.append(f"\n👥 Multi-Agent System")
            self.output.append(f"Roles: {roles}")
            self.output.append(f"Task: {mas_task}")
            self.output.append(f"\n🔄 Would use CrewAI/AutoGen to orchestrate {len(roles)} agents.")

        self.output.append(f"\n{'='*50}")
        self.output.append(f"⚙️ Execution Plan:")
        self.output.append(f"1. Parse goal into subtasks")
        self.output.append(f"2. Select appropriate tools for each subtask")
        self.output.append(f"3. Execute tool calls (MCP protocol)")
        self.output.append(f"4. Synthesize results")
        self.output.append(f"5. Return final output")

        if "ReAct" in arch:
            self.output.append(f"\n🧠 ReAct Loop: Thought → Action → Observation → Thought → ...")
            self.output.append(f"Example: 'Thought: I need to analyze this data. Action: Use Code Interpreter. Observation: ...'")

        if "MCP" in arch:
            self.output.append(f"\n🔌 MCP Server: Agent connects to tools via Model Context Protocol")
            self.output.append(f"Clients: Claude Desktop, Continue.dev, Zed")

        if "A2A" in arch:
            self.output.append(f"\n🔗 A2A Protocol: Secure agent-to-agent communication")
            self.output.append(f"Agent Card discovery, task negotiation, result sharing")

        self.output.append(f"\n{'='*50}")
        self.output.append(f"⚠️ Full execution requires:")
        self.output.append(f"- langchain / CrewAI / AutoGen")
        self.output.append(f"- LLM API key (OpenAI, Anthropic, etc.)")
        self.output.append(f"- MCP SDK (if using MCP)")
        self.output.append(f"\nInstall: pip install langchain crewai autogen mcp")

        self.settings.update("agentic", "config", self.agent_config)