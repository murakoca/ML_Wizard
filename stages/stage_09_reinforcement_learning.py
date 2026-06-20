from stages.stage_base import StageWidget
from PyQt5.QtWidgets import (QLabel, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QMessageBox)
import numpy as np

class Stage09RL(StageWidget):
    def __init__(self, settings):
        super().__init__(settings, "Reinforcement Learning")
        self.set_theory("<h2>09. Reinforcement Learning</h2><p>MDP, Q-Learning, Policy Gradients, PPO, RLHF (Human Feedback), GRPO (DeepSeek), Agent Optimization.</p>")
        self._build_ui()
        self.add_navigation(next_enabled=True)

    def _build_ui(self):
        # Algoritma seçimi
        algo_gb = QGroupBox("1. RL Algorithm")
        algo_vbox = QVBoxLayout()
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self.algo_cb = QComboBox()
        self.algo_cb.addItems([
            "Q-Learning (Tabular)",
            "Deep Q-Network (DQN)",
            "Policy Gradient (REINFORCE)",
            "PPO (Proximal Policy Optimization)",
            "RLHF (Reinforcement Learning from Human Feedback)",
            "GRPO (Group Relative Policy Optimization)"
        ])
        self.algo_cb.currentTextChanged.connect(self._on_algo_change)
        algo_row.addWidget(self.algo_cb)
        algo_vbox.addLayout(algo_row)

        # Environment
        env_row = QHBoxLayout()
        env_row.addWidget(QLabel("Environment:"))
        self.env_cb = QComboBox()
        self.env_cb.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "LunarLander-v2",
            "Custom GridWorld",
            "LLM Fine-Tuning (RLHF)"
        ])
        env_row.addWidget(self.env_cb)
        algo_vbox.addLayout(env_row)
        algo_gb.setLayout(algo_vbox)
        self.layout.addWidget(algo_gb)

        # Parametreler
        param_gb = QGroupBox("2. Hyperparameters")
        param_vbox = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Episodes:"))
        self.episodes = QSpinBox()
        self.episodes.setRange(10, 10000)
        self.episodes.setValue(500)
        row1.addWidget(self.episodes)
        row1.addWidget(QLabel("Learning Rate:"))
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.0001, 0.1)
        self.lr.setSingleStep(0.0001)
        self.lr.setDecimals(4)
        self.lr.setValue(0.001)
        row1.addWidget(self.lr)
        param_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gamma (Discount):"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.8, 0.999)
        self.gamma.setSingleStep(0.01)
        self.gamma.setValue(0.99)
        row2.addWidget(self.gamma)
        row2.addWidget(QLabel("Epsilon:"))
        self.epsilon = QDoubleSpinBox()
        self.epsilon.setRange(0.01, 1.0)
        self.epsilon.setSingleStep(0.05)
        self.epsilon.setValue(0.1)
        row2.addWidget(self.epsilon)
        param_vbox.addLayout(row2)

        param_gb.setLayout(param_vbox)
        self.layout.addWidget(param_gb)

        # RLHF özel seçenekleri (gizli)
        self.rlhf_gb = QGroupBox("RLHF / GRPO Options")
        rlhf_vbox = QVBoxLayout()
        rlhf_vbox.addWidget(QLabel("Reward Model:"))
        self.reward_cb = QComboBox()
        self.reward_cb.addItems(["Human Preference", "Automated Metric", "Toxicity Score", "Helpfulness Score"])
        rlhf_vbox.addWidget(self.reward_cb)
        self.kl_check = QCheckBox("Use KL Divergence Penalty")
        self.kl_check.setChecked(True)
        rlhf_vbox.addWidget(self.kl_check)
        self.rlhf_gb.setLayout(rlhf_vbox)
        self.rlhf_gb.hide()
        self.layout.addWidget(self.rlhf_gb)

        # Run
        self.run_btn = QPushButton("3. Run RL Training")
        self.run_btn.clicked.connect(self.run_rl)
        self.layout.addWidget(self.run_btn)

        # Çıktı
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)

    def _on_algo_change(self, algo):
        if "RLHF" in algo or "GRPO" in algo:
            self.rlhf_gb.show()
        else:
            self.rlhf_gb.hide()

    def run_rl(self):
        algo = self.algo_cb.currentText()
        env_name = self.env_cb.currentText()
        episodes = self.episodes.value()
        lr = self.lr.value()
        gamma = self.gamma.value()
        epsilon = self.epsilon.value()

        self.output.clear()
        self.output.append(f"🎮 RL Training Configuration\n{'='*50}")
        self.output.append(f"Algorithm: {algo}")
        self.output.append(f"Environment: {env_name}")
        self.output.append(f"Episodes: {episodes}")
        self.output.append(f"Learning Rate: {lr}")
        self.output.append(f"Gamma: {gamma}")
        self.output.append(f"Epsilon: {epsilon}")
        self.output.append(f"\n{'='*50}")

        if "Q-Learning" in algo and "Tabular" in algo:
            self.output.append(f"\n📊 Tabular Q-Learning Simulation")
            # Basit Q-Learning simülasyonu
            n_states = 25  # 5x5 grid
            n_actions = 4
            Q = np.zeros((n_states, n_actions))
            rewards_history = []

            for ep in range(min(episodes, 100)):  # Demo için sınırlı
                state = 0
                total_reward = 0
                for step in range(50):
                    if np.random.random() < epsilon:
                        action = np.random.randint(n_actions)
                    else:
                        action = np.argmax(Q[state])
                    next_state = min(max(state + np.random.choice([-1, 1, -5, 5]), 0), n_states-1)
                    reward = 1.0 if next_state == n_states-1 else -0.1
                    Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
                    state = next_state
                    total_reward += reward
                rewards_history.append(total_reward)

            self.output.append(f"Final Q-Table shape: {Q.shape}")
            self.output.append(f"Average reward (last 10): {np.mean(rewards_history[-10:]):.3f}")
            self.output.append(f"Max reward: {np.max(rewards_history):.3f}")

        elif "RLHF" in algo:
            self.output.append(f"\n🧑‍🏫 RLHF Training Pipeline")
            self.output.append(f"1. Supervised Fine-Tuning (SFT)")
            self.output.append(f"2. Reward Model Training")
            reward_model = self.reward_cb.currentText()
            self.output.append(f"   - Using: {reward_model}")
            self.output.append(f"3. PPO Optimization")
            if self.kl_check.isChecked():
                self.output.append(f"   - KL Penalty: Enabled (β=0.1)")
            self.output.append(f"4. Final Policy Model")
            self.output.append(f"\n⚠️ Requires: trl, transformers, datasets")
            self.output.append(f"Install: pip install trl accelerate")

        elif "GRPO" in algo:
            self.output.append(f"\n🔄 GRPO (Group Relative Policy Optimization)")
            self.output.append(f"Used by: DeepSeek-R1")
            self.output.append(f"Key Features:")
            self.output.append(f"- Group-based advantage estimation")
            self.output.append(f"- Relative comparisons within groups")
            self.output.append(f"- No separate value model needed")
            self.output.append(f"\nAdvantage over PPO: Simpler, more stable training")

        else:
            self.output.append(f"\n🤖 Would use Gymnasium + Stable-Baselines3")
            self.output.append(f"Install: pip install gymnasium stable-baselines3")
            self.output.append(f"\nSample code:")
            self.output.append(f"import gymnasium as gym")
            self.output.append(f"from stable_baselines3 import PPO")
            self.output.append(f"env = gym.make('{env_name}')")
            self.output.append(f"model = PPO('MlpPolicy', env, learning_rate={lr})")
            self.output.append(f"model.learn(total_timesteps={episodes})")

        self.settings.update("rl", "algorithm", algo)
        self.settings.update("rl", "environment", env_name)
        self.settings.update("rl", "episodes", episodes)