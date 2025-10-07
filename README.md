# Autonomous Multi-Goal Drone Navigation using Deep Reinforcement Learning

## Project Overview

This project implements a deep reinforcement learning (DRL) agent to train a drone for autonomous navigation in a 2D simulated environment. The goal is to train the drone to visit multiple target locations in sequence while avoiding obstacles.

The project uses the Proximal Policy Optimization (PPO) algorithm, a state-of-the-art reinforcement learning technique, implemented with Python and the PyTorch library. To make the learning process more efficient, the project also incorporates Curriculum Learning, where the agent is first trained in simple environments and then gradually exposed to more complex scenarios.

## Core Technologies

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **RL Algorithm:** Proximal Policy Optimization (PPO)
- **Training Strategy:** Curriculum Learning
- **Environment:** Custom 2D simulation built with Gymnasium (formerly OpenAI Gym)

## How to Run

1. Ensure Python 3 and required packages (`torch`, `numpy`, `gymnasium`, `matplotlib`) are installed.
2. Run the main training script:
   ```bash
   python drone_nav.py
   ```
3. Follow the on-screen prompts to enable or disable training visualizations.
4. After training, the model will be saved to the `models/` directory, and an evaluation will be run.

Other scripts:
- `watch_drone.py`: Interactively select and watch a pre-trained model navigate the environment.
- `resume_training.py`: Resume training from a saved model checkpoint.

## Development Journey: Challenges and Changes

This project evolved significantly from its initial version. The primary challenge was training the agent to be both successful and efficient, leading to several key improvements.

### Initial Challenges

- **Inefficient Paths:** The agent learned to reach goals but often took meandering, suboptimal paths.
- **Jerky Movements:** The drone's movements were often erratic and unnatural, indicating a need for smoother control.
- **Slow Learning:** The agent required many episodes to learn basic behaviors, especially in complex environments.
- **Incorrect Code Implementation:** Initial development was blocked by a `TypeError` due to a misunderstanding of how class constructors were defined and called.

### Key Changes and Improvements (V1.1)

To address these challenges, the following changes were implemented:

1. **Enhanced Reward Shaping:**
   - A **directional alignment bonus** was added to reward the drone for flying directly towards its target, promoting smoother, more direct flight paths.
   - The **time penalty** was increased to encourage the agent to find solutions more quickly and efficiently.

2. **Hyperparameter Tuning:**
   - The **learning rate** and **entropy coefficient** were carefully adjusted to balance exploration and exploitation, leading to more stable training.

3. **Smoother Curriculum:**
   - The curriculum was expanded from 3 to 6 stages, providing a more gradual increase in difficulty. This helped the agent generalize better from simple to complex tasks.

4. **Completed Evaluation Logic:**
   - The `evaluate()` function was fully implemented to provide clear, quantitative metrics on the final model's performance, such as success rate and average reward.

5. **Dynamic Model Management:**
   - All scripts now prompt for model filenames when saving or loading, with defaults provided.
   - `watch_drone.py` lists available models in `models/` and prompts the user to select one for visualization.

6. **Enhanced Network Architecture:**
   - Hidden layers increased from 256 to 512 units.
   - Added shared feature extractor, dropout for regularization, and separate actor and critic heads.

7. **Updated Hyperparameters:**
   - LR: `3e-4`, γ: `0.995`, λ: `0.98`, entropy coef: `0.01`, batch size: `128`, buffer size: `4096`.

8. **Advanced Reward Shaping:**
   - Stronger distance and velocity alignment bonuses, speed-efficiency rewards, and progressive goal completion bonuses.
   - Reduced time penalty and smoother acceleration penalties.

9. **Backward Compatibility Loader:**
   - Legacy 2×256 architecture loader with optional weight transfer to the enhanced model.

10. **Bug Fixes:**
    - Resolved backward graph error in PPO update by detaching tensors.
    - Improved GAE computation and terminal state handling.

### Changelog

#### 2025-10-07
- **Interactive Model Selection:** `watch_drone.py` now lists and prompts for available `.pth` files.
- **Unified CLI:** All scripts (`drone_nav.py`, `resume_training.py`, `watch_drone.py`) support dynamic model naming and loading.
- **Network Architecture:** Upgraded to 512-unit hidden layers with dropout and separate heads.
- **Hyperparameter Updates:** Faster learning and improved exploration.
- **Reward Shaping:** Enhanced shaping for smoother, more efficient paths.
- **Curriculum Learning:** Six gradual stages with success-rate thresholds.
- **Compatibility:** Legacy model support with optional weight transfer.
- **Bug Fixes:** PPO graph error, GAE improvements.
- **Usability:** Clear prompts and error messages.

## License


