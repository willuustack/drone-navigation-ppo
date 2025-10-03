# Autonomous Multi-Goal Drone Navigation

**PPO + Curriculum Learning Implementation for Intelligent 2D Navigation**

An autonomous drone system that learns to navigate complex environments using deep reinforcement learning. The agent masters obstacle avoidance, multi-goal path planning, and efficient navigation through progressive curriculum training.

## Project Overview

This project implements a sophisticated drone navigation system using:
- **Proximal Policy Optimization (PPO)** for stable reinforcement learning
- **Curriculum Learning** with progressive difficulty stages
- **Lidar-based perception** for real-time obstacle detection
- **Physics simulation** with realistic inertia, drag, and movement dynamics
- **Real-time visualization** showing live drone navigation

### Key Achievements
-  **98%+ success rate** in complex navigation scenarios
-  **Multi-goal planning** with intelligent route optimization
-  **Real-time obstacle avoidance** using 8-ray lidar simulation
-  **Smooth trajectory generation** with physics-based movement
-  **Curriculum progression** from basic to advanced scenarios

## Features

### Intelligent Navigation
- **Multi-goal sequential visiting** with dynamic target selection
- **Collision-free path planning** in obstacle-rich environments
- **Adaptive speed control** based on environment complexity
- **Efficient route optimization** minimizing travel distance

### Advanced Training
- **Curriculum Learning**: Progressive difficulty from 2→6 obstacles, 3→5 goals
- **Custom reward shaping** encouraging smooth, efficient paths
- **Generalized Advantage Estimation (GAE)** for stable learning
- **Automatic curriculum advancement** based on performance metrics

### Realistic Simulation
- **Physics-based movement** with configurable inertia and drag
- **8-ray lidar sensor** for 360° obstacle detection
- **Customizable drone parameters** (speed, acceleration, turning rate)
- **Dynamic obstacle generation** for varied training scenarios

## Live Visualization

Watch your trained drone navigate in real-time with:
- **Animated drone movement** (blue circle with trail)
- **Dynamic lidar rays** (red lines showing sensor readings)
- **Goal progression** (red → green as goals are reached)
- **Obstacle visualization** (gray circles of varying sizes)

## Installation & Usage

### Prerequisites
Install required packages
pip install torch torchvision gymnasium numpy matplotlib

### Quick Start
Clone the repository
git clone https://github.com/willuustack/drone-navigation-ppo.git
cd drone-navigation-ppo

Train a new model (10-15 minutes)
python3 drone_nav.py

Watch your trained drone navigate
python3 watch_drone.py

### Training Options
Interactive configuration with custom parameters
python3 drone_nav.py

Answer prompts to customize:
- Drone physics (speed, inertia, drag)
- Environment complexity (obstacles, goals)
- Training duration
Training with visualization (see progress every 100 episodes)
Choose 'y' when prompted for training visualization
text

## Configuration

### Drone Physics Parameters
max_speed: 3.0 # Maximum drone velocity
inertia_factor: 0.85 # Movement smoothness (0=instant, 1=drift)
drag_coefficient: 0.3 # Air resistance simulation
mass: 1.0 # Affects acceleration response



### Environment Settings
num_obstacles: 5 # Obstacle density
obstacle_size: 0.5-1.5 # Obstacle radius range
num_goals: 5 # Sequential goals to visit
lidar_range: 3.0 # Sensor detection distance



### Training Hyperparameters
learning_rate: 3e-4 # PPO optimization rate
gamma: 0.99 # Discount factor
batch_size: 64 # Training batch size
buffer_size: 2048 # Experience replay buffer



## Training Results

### Curriculum Progression
| Stage | Obstacles | Goals | Max Speed | Success Rate |
|-------|-----------|--------|-----------|-------------|
| 1 (Easy) | 2 | 3 | 2.0 | 80%+ |
| 2 (Medium) | 4 | 4 | 2.5 | 80%+ |
| 3 (Hard) | 6 | 5 | 3.0 | 80%+ |

### Performance Metrics
- **Training Episodes**: ~1000 episodes total
- **Convergence Time**: 10-15 minutes on M4 MacBook
- **Final Success Rate**: 98%+ in complex scenarios
- **GPU Acceleration**: Automatic MPS/CUDA detection

##  Architecture

### Neural Network
- **Input**: 21-dimensional state space (position, velocity, goals, lidar)
- **Hidden Layers**: 256→256 fully connected with ReLU
- **Output**: Continuous 2D acceleration commands
- **Optimization**: Adam optimizer with gradient clipping

### State Representation
- Drone position and velocity (4D)
- Target goal vector and distance (4D) 
- Goal completion status (5D binary)
- Lidar sensor readings (8D normalized)

### Reward Design
- **Goal completion**: +50 per goal, +100 for all goals
- **Collision penalty**: -100 (episode termination)
- **Distance reward**: Encourages goal approach
- **Alignment reward**: Promotes direct paths
- **Time penalty**: Encourages efficiency

##  Project Structure

drone-navigation-ppo/
├── drone_nav.py # Main training script with PPO implementation
├── watch_drone.py # Real-time visualization of trained agent
├── drone_navigation_ppo.pth # Saved trained model (3.2MB)
└── README.md # This file



##  Educational Context

Developed for **MECH 296: Introduction to Reinforcement Learning** at Santa Clara University. This project demonstrates:

- **Deep Reinforcement Learning** implementation from scratch
- **Curriculum Learning** for complex skill acquisition  
- **Multi-objective optimization** in continuous action spaces
- **Real-time visualization** of learned behaviors
- **Professional software development** practices

##  Technical Implementation

### PPO Algorithm Features
- **Clipped surrogate objective** for stable policy updates
- **Generalized Advantage Estimation** for variance reduction
- **Entropy regularization** preventing premature convergence
- **Value function learning** for better state evaluation

### Curriculum Learning Strategy
- **Automatic progression** based on 80% success rate threshold
- **Gradual complexity increase** in obstacles and goals
- **Preserved learning** across curriculum stages
- **Robust final performance** in diverse scenarios

##  Future Enhancements

- [ ] **3D navigation** with altitude control
- [ ] **Dynamic obstacles** with moving entities
- [ ] **Multi-agent scenarios** with drone swarms
- [ ] **Real drone deployment** using ROS integration
- [ ] **Advanced sensors** (camera-based vision)


##  Author

**William Lu** - Reinforcement Learning Implementation  

---
