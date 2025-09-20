#♠️ Texas Mayhem – AI-Powered Poker Engine


![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-100%25-success.svg)

A complete **Heads-Up Texas Hold’em Poker Engine** with built-in **AI opponents**, including:
- Rule-based **Heuristic AI**  
- **NumPy REINFORCE Agent**  
- **PyTorch A2C Agent** (Actor-Critic)  

The project blends **game mechanics, probability simulations, and reinforcement learning** into a modular, test-driven framework for studying decision-making under uncertainty.

---

## 🎯 Features

- **Game Engine**
  - Full 52-card deck with Unicode/ASCII suits.
  - Hand evaluator (quads, flush, full house, ace-low straights).
  - Betting rounds: preflop, flop, turn, river.
  - Blind structure & stack management.

- **AI Opponents**
  - **HeuristicAI** – Monte Carlo equity vs pot odds decision-making.
  - **RLAgent (NumPy)** – lightweight policy gradient learner.
  - **TorchA2CAgent (PyTorch)** – advanced actor-critic with entropy regularization & replay buffer.

- **Learning & Training**
  - Dataset generation from heuristic self-play.
  - Imitation learning (NumPy / PyTorch).
  - Self-play tournaments for reinforcement learning.
  - Grid search & Bayesian optimization for hyperparameter tuning.

- **Analytics & Logging**
  - Logs every hand, action, and final reward.
  - Compute VPIP, PFR, Aggression Factor.
  - CSV export for Excel/Tableau.
  - Basic charts (cumulative winnings, action histograms).

- **CLI Gameplay**
  - Human vs AI mode.
  - Interactive betting prompts.
  - AI-powered advice mode (equity, draws, pot odds).
  - Full rules walkthrough for beginners.

- **Testing**
  - 15+ unit test categories for cards, deck, hand evaluation, equity, AI decisions, dataset encoding, logging.

---

## 🏗️ Architecture


Key components:

- `Card`, `Deck` → Poker fundamentals.  
- `HandEvaluator` → Evaluates 7 cards → best 5-card hand.  
- `HeuristicAI` → Rule-based decisions with Monte Carlo simulations.  
- `RLAgent`, `MLPPolicy` → NumPy reinforcement learning.  
- `TorchA2CAgent`, `TorchPolicy` → PyTorch actor-critic agent.  
- `generate_heuristic_dataset` → Collects labeled states for imitation learning.  
- `train_imitation_numpy`, `train_imitation_torch` → Supervised imitation.  
- `selfplay_train_torch` → Torch self-play training.  
- `analytics_report`, `plot_basic_charts` → Player stats & visualization.  

---

## 🧑‍💻 Installation

### Requirements
- Python 3.9+
- Recommended libraries:
  ```bash
  pip install numpy pandas matplotlib torch


Usage

python Texas_Mayhem_Version_V2.py

1) Play vs Heuristic AI
2) Play vs NumPy RL Agent
3) Train NumPy imitation model, then play
4) Play vs PyTorch A2C (if torch installed)
5) Train PyTorch imitation model, then play
6) PyTorch self-play tournament (optional CSV log)
7) Torch grid search
8) Torch Bayesian hyperparameter search


Train Agents:

NumPy imitation learning:

from Texas_Mayhem_Version_V2 import MLPPolicy, train_imitation_numpy
policy = MLPPolicy(input_dim=100, hidden=64)
train_imitation_numpy(policy, epochs=3, hands_per_epoch=300)

PyTorch imitation learning:

from Texas_Mayhem_Version_V2 import TorchPolicy, train_imitation_torch
import torch
policy = TorchPolicy(input_dim=100, hidden=128)
train_imitation_torch(policy, device="cpu", epochs=3)

Self-play training (Torch A2C vs Heuristic):

from Texas_Mayhem_Version_V2 import selfplay_train_torch
policy = selfplay_train_torch(episodes=500)

3. Analytics

After a session, choose to save hand history to CSV:
hand_history.csv → detailed per-action logs.
tourney_log.csv → tournament results (self-play mode).
Optionally generate plots:
Cumulative winnings by player.
Action frequency histogram.

Covers:

Deck completeness.
Hand evaluator edge cases (wheel straight, kickers).
Monte Carlo equity tests.
AI decision-making.
Action mapping correctness.
State encoding consistency.
Logging & reward backfill.


🚀 Roadmap / Future Work

Add multi-player table support (not just heads-up).
Optimize Monte Carlo simulations with parallelism.
Implement Deep RL architectures (e.g., PPO, DQN).
Build a GUI/Web frontend.
Integrate with online datasets (hand histories from real games).

🙌 Credits

Author: Vrajesh Shah
Core Libraries: Python, NumPy, PyTorch, Pandas, Matplotlib
Inspiration: AI research on poker (DeepStack, Libratus)

