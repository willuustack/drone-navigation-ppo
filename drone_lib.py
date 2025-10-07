"""
Autonomous Multi-Goal Drone Navigation - V1.1
PPO + Curriculum Learning with enhanced reward shaping and hyperparameter tuning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import sys, matplotlib

# ----- Cross-platform Matplotlib backend -----
if sys.platform == "darwin":
    matplotlib.use("MacOSX")  # macOS GUI backend
elif sys.platform.startswith("win"):
    matplotlib.use("TkAgg")  # Windows GUI backend (no extra installs)
else:
    matplotlib.use("Agg")  # Headless-safe fallback (saves figures)
# --------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import os
from dataclasses import dataclass
from typing import List, Tuple

# ==================== CONFIGURATION ====================

@dataclass
class DroneConfig:
    max_speed: float = 2.5
    max_acceleration: float = 2.0
    drag_coefficient: float = 0.4
    inertia_factor: float = 0.90
    mass: float = 1.0

@dataclass
class EnvironmentConfig:
    map_size: Tuple[float, float] = (10.0, 10.0)
    num_obstacles: int = 5
    obstacle_size_range: Tuple[float, float] = (0.5, 1.5)
    num_goals: int = 5
    lidar_rays: int = 16
    lidar_range: float = 4.0
    goal_reach_threshold: float = 0.35
    collision_threshold: float = 0.2

@dataclass
class TrainingConfig:
    # Enhanced hyperparameters for better performance
    learning_rate: float = 3e-4  # Increased for faster learning
    gamma: float = 0.995  # Increased for longer horizon
    gae_lambda: float = 0.98  # Increased from 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01  # Increased for more exploration
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4  # Reduced to prevent overfitting
    batch_size: int = 128  # Increased for better gradient estimates
    buffer_size: int = 4096  # Increased for more diverse experience

@dataclass
class CurriculumConfig:
    stages: List[dict] = None

    def __post_init__(self):
        if self.stages is None:
            # Smoother curriculum progression
            self.stages = [
                {"num_obstacles": 1, "max_speed": 1.5, "num_goals": 2},  # Start easier
                {"num_obstacles": 2, "max_speed": 1.8, "num_goals": 3},
                {"num_obstacles": 3, "max_speed": 2.0, "num_goals": 3},
                {"num_obstacles": 4, "max_speed": 2.2, "num_goals": 4},
                {"num_obstacles": 5, "max_speed": 2.3, "num_goals": 4},
                {"num_obstacles": 6, "max_speed": 2.5, "num_goals": 5},  # Final stage
            ]

# ==================== ENVIRONMENT ====================

class DroneNavigationEnv(gym.Env):
    def __init__(self, drone_config: DroneConfig, env_config: EnvironmentConfig, max_goals: int):
        super().__init__()
        self.drone_config = drone_config
        self.env_config = env_config
        self.max_goals = max_goals

        state_dim = 8 + max_goals + env_config.lidar_rays
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.drone_pos = np.zeros(2)
        self.drone_vel = np.zeros(2)
        self.steps = 0
        self.max_steps = 1500
        self.prev_acceleration = np.zeros(2)
        self.position_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.random.uniform(1, self.env_config.map_size[0] - 1, size=2)
        self.drone_vel = np.zeros(2)
        self.obstacles = self._generate_obstacles()
        self.goals = self._generate_goals()
        self.goals_visited = [False] * len(self.goals)
        self.current_goal_idx = 0
        self.steps = 0
        self.prev_acceleration = np.zeros(2)
        self.position_history = [self.drone_pos.copy()]
        return self._get_observation(), {}

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.env_config.num_obstacles):
            while True:
                pos = np.random.uniform(0.5, self.env_config.map_size[0] - 0.5, size=2)
                radius = np.random.uniform(*self.env_config.obstacle_size_range)
                if np.linalg.norm(pos - self.drone_pos) > radius + 1.5:
                    obstacles.append((pos, radius))
                    break
        return obstacles

    def _generate_goals(self):
        goals = []
        for _ in range(self.env_config.num_goals):
            while True:
                pos = np.random.uniform(1, self.env_config.map_size[0] - 1, size=2)
                if all(np.linalg.norm(pos - obs_pos) >= obs_radius + 0.5 for obs_pos, obs_radius in self.obstacles):
                    goals.append(pos)
                    break
        return goals

    def _get_lidar_readings(self) -> np.ndarray:
        readings = np.full(self.env_config.lidar_rays, self.env_config.lidar_range)
        angles = np.linspace(0, 2 * np.pi, self.env_config.lidar_rays, endpoint=False)

        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)])
            for obs_pos, obs_radius in self.obstacles:
                to_obstacle = obs_pos - self.drone_pos
                b = np.dot(to_obstacle, direction)
                c = np.dot(to_obstacle, to_obstacle) - obs_radius**2
                delta = b**2 - c
                if delta >= 0:
                    t = b - np.sqrt(delta)
                    if 0 < t < readings[i]:
                        readings[i] = t
        return readings / self.env_config.lidar_range

    def _get_observation(self) -> np.ndarray:
        unvisited_goals = [i for i, visited in enumerate(self.goals_visited) if not visited]
        if unvisited_goals:
            distances = [np.linalg.norm(self.goals[i] - self.drone_pos) for i in unvisited_goals]
            self.current_goal_idx = unvisited_goals[np.argmin(distances)]

        current_goal = self.goals[self.current_goal_idx]
        to_goal = current_goal - self.drone_pos
        goal_dist = np.linalg.norm(to_goal)
        goal_angle = np.arctan2(to_goal[1], to_goal[0])

        lidar = self._get_lidar_readings()
        goals_visited_padded = np.array(self.goals_visited + [False] * (self.max_goals - len(self.goals_visited)), dtype=np.float32)

        return np.concatenate([
            self.drone_pos / self.env_config.map_size[0],
            self.drone_vel / self.drone_config.max_speed,
            to_goal / self.env_config.map_size[0],
            [goal_dist / self.env_config.map_size[0], goal_angle / np.pi],
            goals_visited_padded, lidar
        ]).astype(np.float32)

    def _check_collision(self) -> bool:
        if not (0 < self.drone_pos[0] < self.env_config.map_size[0] and 0 < self.drone_pos[1] < self.env_config.map_size[1]):
            return True
        return any(np.linalg.norm(self.drone_pos - p) < r + self.env_config.collision_threshold for p, r in self.obstacles)

    def step(self, action: np.ndarray):
        self.steps += 1
        acceleration = action * self.drone_config.max_acceleration
        self.drone_vel = self.drone_vel * self.drone_config.inertia_factor + acceleration / self.drone_config.mass

        # Apply drag
        if np.linalg.norm(self.drone_vel) > 0:
            self.drone_vel -= self.drone_config.drag_coefficient * self.drone_vel * 0.1

        # Speed limit
        if np.linalg.norm(self.drone_vel) > self.drone_config.max_speed:
            self.drone_vel = self.drone_vel / np.linalg.norm(self.drone_vel) * self.drone_config.max_speed

        self.drone_pos += self.drone_vel * 0.1

        self.position_history.append(self.drone_pos.copy())
        if len(self.position_history) > 2:
            self.position_history.pop(0)

        terminated = False
        reward = 0.0

        if self._check_collision():
            reward = -100.0
            terminated = True
        else:
            current_goal = self.goals[self.current_goal_idx]
            dist_to_goal = np.linalg.norm(current_goal - self.drone_pos)

            # Enhanced reward shaping
            if len(self.position_history) >= 2:
                prev_dist = np.linalg.norm(current_goal - self.position_history[-2])
                reward += (prev_dist - dist_to_goal) * 20.0  # Increased from 10.0

            # Enhanced velocity alignment bonus
            vel_norm = np.linalg.norm(self.drone_vel)
            if vel_norm > 1e-4:
                vel_dir = self.drone_vel / vel_norm
                to_goal_dir = (current_goal - self.drone_pos) / dist_to_goal
                alignment = np.dot(vel_dir, to_goal_dir)
                reward += alignment * 8.0  # Increased from 5.0

            # Speed bonus for efficient movement
            optimal_speed = min(2.0, dist_to_goal * 2.0)
            speed_efficiency = 1.0 - abs(vel_norm - optimal_speed) / self.drone_config.max_speed
            reward += speed_efficiency * 3.0

            # Goal reaching rewards
            if dist_to_goal < self.env_config.goal_reach_threshold:
                if not self.goals_visited[self.current_goal_idx]:
                    reward += 100.0  # Increased from 50.0
                    self.goals_visited[self.current_goal_idx] = True

                    # Progressive bonus for multiple goals
                    goals_completed = sum(self.goals_visited)
                    reward += goals_completed * 25.0  # Bonus scaling

                    if all(self.goals_visited):
                        reward += 200.0  # Increased from 100.0
                        terminated = True

            # Reduced time penalty for longer episodes
            reward -= 0.1  # Reduced from 0.2

            # Smoothness reward (reduce jerkiness)
            accel_change = np.linalg.norm(acceleration - self.prev_acceleration)
            reward -= accel_change * 0.02  # Reduce penalty
            self.prev_acceleration = acceleration.copy()

        if self.steps >= self.max_steps:
            terminated = True

        return self._get_observation(), reward, terminated, False, {}

# ==================== PPO AGENT ====================

# Legacy ActorCritic for backward compatibility with old models
class LegacyActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist, value

# New Enhanced ActorCritic for better performance
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):  # Increased from 256
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate heads for actor and critic
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        value = self.critic(features)
        mu = self.actor(features)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            dist, _ = self.policy(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
        return action.cpu().numpy(), log_prob.item()

    def store_transition(self, s, a, r, s_prime, done, prob_a):
        self.buffer.append((s, a, [r], s_prime, [done], prob_a))

    def update(self):
        if len(self.buffer) == 0:
            return

        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.buffer)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).squeeze().to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).squeeze().to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(self.device)

        # Compute values for all states
        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze()

            # Compute returns using GAE
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)

            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0  # Terminal state
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates - Fixed gradient issue
        for epoch in range(self.config.ppo_epochs):
            # Get fresh forward pass for each epoch
            dist, curr_values = self.policy(states)
            curr_values = curr_values.squeeze()

            # Compute new log probabilities
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()

            # Compute ratio and losses
            ratio = torch.exp(new_log_probs - old_log_probs.detach())  # Detach old_log_probs
            surr1 = ratio * advantages.detach()  # Detach advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(curr_values, returns.detach())  # Detach returns
            total_loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # Clear buffer
        self.buffer = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        try:
            # Try loading with new architecture first
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            print("✅ Loaded model with enhanced architecture")
        except (RuntimeError, KeyError) as e:
            print("⚠️  Model architecture mismatch. Trying legacy compatibility...")
            try:
                # Create legacy model and load old weights
                state_dim = self.policy.shared[0].in_features
                action_dim = self.policy.log_std.shape[0]
                legacy_policy = LegacyActorCritic(state_dim, action_dim).to(self.device)
                legacy_policy.load_state_dict(torch.load(path, map_location=self.device))

                # Transfer weights to new architecture
                self._transfer_weights(legacy_policy)
                print("✅ Successfully transferred legacy model to enhanced architecture")
            except Exception as transfer_error:
                print(f"❌ Failed to load model: {transfer_error}")
                raise transfer_error

    def _transfer_weights(self, legacy_policy):
        """Transfer weights from legacy model to new enhanced architecture"""
        with torch.no_grad():
            # Transfer shared layer weights (from legacy actor first layers)
            self.policy.shared[0].weight.copy_(legacy_policy.actor[0].weight)
            self.policy.shared[0].bias.copy_(legacy_policy.actor[0].bias)
            self.policy.shared[3].weight.copy_(legacy_policy.actor[2].weight)
            self.policy.shared[3].bias.copy_(legacy_policy.actor[2].bias)

            # Transfer actor head weights (from legacy actor last layer)
            self.policy.actor[1].weight.copy_(legacy_policy.actor[4].weight)
            self.policy.actor[1].bias.copy_(legacy_policy.actor[4].bias)

            # Transfer critic head weights (from legacy critic last layer)
            self.policy.critic[1].weight.copy_(legacy_policy.critic[4].weight)
            self.policy.critic[1].bias.copy_(legacy_policy.critic[4].bias)

            # Transfer log_std parameter
            self.policy.log_std.copy_(legacy_policy.log_std)

            print("🔄 Transferred weights from legacy to enhanced architecture")

# ==================== TRAINING SYSTEM ====================

class CurriculumTrainer:
    def __init__(self, drone_config, env_config, training_config, curriculum_config):
        self.env = DroneNavigationEnv(drone_config, env_config, env_config.num_goals)
        self.agent = PPOAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0], training_config)
        self.metrics = {'episode_rewards': [], 'success_rate': []}
        self.curriculum_config = curriculum_config
        self.drone_config_base = drone_config
        self.env_config_base = env_config

    def apply_curriculum_stage(self, stage):
        if stage < len(self.curriculum_config.stages):
            cfg = self.curriculum_config.stages[stage]
            self.env.env_config.num_obstacles = cfg['num_obstacles']
            self.env.drone_config.max_speed = cfg['max_speed']
            self.env.env_config.num_goals = cfg['num_goals']
            print(f"📚 Curriculum Stage {stage + 1}: {cfg}")

    def _quick_render(self, trajectory, episode_num):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Training Path Snapshot - Episode {episode_num}")
        ax.set_xlim(0, self.env.env_config.map_size[0])
        ax.set_ylim(0, self.env.env_config.map_size[1])

        for obs_pos, obs_radius in self.env.obstacles:
            ax.add_patch(Circle(obs_pos, obs_radius, color='gray', alpha=0.7))

        for i, goal in enumerate(self.env.goals):
            color = 'green' if self.env.goals_visited[i] else 'red'
            ax.add_patch(Circle(goal, 0.2, color=color, alpha=0.8))

        path_array = np.array(trajectory)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1.5, alpha=0.8)
        ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=10, label='Start')
        ax.plot(path_array[-1, 0], path_array[-1, 1], 'rx', markersize=10, label='End')
        ax.legend()
        ax.grid(True)
        plt.show()

    def train_with_visualization(self, total_episodes=3000, eval_interval=10, render_every=100):
        print("🚀 Starting Enhanced Training Loop with Visualization...")
        recent_successes = []
        current_stage = 0
        self.apply_curriculum_stage(current_stage)

        for episode in range(1, total_episodes + 1):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            trajectory = [self.env.drone_pos.copy()]

            while not done:
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.store_transition(state, action, reward, next_state, done, log_prob)
                state = next_state
                episode_reward += reward
                trajectory.append(self.env.drone_pos.copy())

            if len(self.agent.buffer) >= self.agent.config.buffer_size:
                self.agent.update()

            if episode % render_every == 0:
                print(f"\n🎬 Visualizing path for Episode {episode}...")
                self._quick_render(trajectory, episode)

            recent_successes.append(1 if all(self.env.goals_visited) else 0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)

            self.metrics['episode_rewards'].append(episode_reward)

            if episode % eval_interval == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
                success_rate = np.mean(recent_successes)
                print(f"📊 Episode {episode}/{total_episodes} | Stage: {current_stage+1} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1%}")

            # More frequent and stricter curriculum advancement
            if episode % 50 == 0 and len(recent_successes) >= 50 and np.mean(recent_successes[-50:]) > 0.85:
                if current_stage < len(self.curriculum_config.stages) - 1:
                    current_stage += 1
                    self.apply_curriculum_stage(current_stage)
                    recent_successes = []

    def train(self, total_episodes=3000, eval_interval=10):
        print("🚀 Starting Enhanced Training Loop...")
        recent_successes = []
        current_stage = 0
        self.apply_curriculum_stage(current_stage)

        for episode in range(1, total_episodes + 1):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.store_transition(state, action, reward, next_state, done, log_prob)
                state = next_state
                episode_reward += reward

            if len(self.agent.buffer) >= self.agent.config.buffer_size:
                self.agent.update()

            recent_successes.append(1 if all(self.env.goals_visited) else 0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)

            self.metrics['episode_rewards'].append(episode_reward)

            if episode % eval_interval == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
                success_rate = np.mean(recent_successes) if recent_successes else 0.0
                print(f"📊 Episode {episode}/{total_episodes} | Stage: {current_stage+1} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.1%}")

            # More frequent and stricter curriculum advancement
            if episode % 50 == 0 and len(recent_successes) >= 50 and np.mean(recent_successes[-50:]) > 0.85:
                if current_stage < len(self.curriculum_config.stages) - 1:
                    current_stage += 1
                    self.apply_curriculum_stage(current_stage)
                    recent_successes = []

    def save_model(self, model_name="drone_navigation_balanced.pth"):
        os.makedirs("models", exist_ok=True)
        save_path = os.path.join("models", model_name)
        self.agent.save(save_path)
        print(f"\n💾 Enhanced model saved to {save_path}")

    def evaluate(self, num_episodes=5, render=False):
        print(f"\n📊 Evaluating for {num_episodes} episodes...")
        total_rewards = []
        success_count = 0

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            total_rewards.append(episode_reward)
            if all(self.env.goals_visited):
                success_count += 1

        avg_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes
        print(f"✅ Average Reward: {avg_reward:.2f}")
        print(f"🎯 Success Rate: {success_rate:.1%}")

    def animate_episode(self, num_episodes=5, interval=40):
        """
        Runs several episodes and creates a smooth Matplotlib animation for each one.
        """
        print(f"\n🎬 Animating {num_episodes} evaluation episodes...")

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            trajectory = [self.env.drone_pos.copy()]

            while not done:
                action, _ = self.agent.select_action(state)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                trajectory.append(self.env.drone_pos.copy())

            # Create animation
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title(f"Enhanced Drone Navigation - Episode {ep + 1}")
            ax.set_xlim(0, self.env.env_config.map_size[0])
            ax.set_ylim(0, self.env.env_config.map_size[1])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Draw obstacles
            for obs_pos, obs_radius in self.env.obstacles:
                ax.add_patch(Circle(obs_pos, obs_radius, color='gray', alpha=0.8, zorder=5))

            # Draw goals
            for i, goal in enumerate(self.env.goals):
                color = 'lime' if self.env.goals_visited[i] else 'tomato'
                ax.add_patch(Circle(goal, 0.2, color=color, alpha=0.9, zorder=10))

            path_array = np.array(trajectory)
            drone_path, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, zorder=20)
            drone_marker, = ax.plot([], [], 'ro', markersize=8, zorder=21, label="Drone")

            ax.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=12, zorder=15, label='Start')

            def update(frame):
                drone_path.set_data(path_array[:frame, 0], path_array[:frame, 1])
                drone_marker.set_data([path_array[frame, 0]], [path_array[frame, 1]])
                return drone_path, drone_marker

            ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=True, repeat=False)
            ax.legend()
            plt.show()