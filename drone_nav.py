"""
Autonomous Multi-Goal Drone Navigation using PPO + Curriculum Learning
Complete implementation with real-time visualization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import os
from dataclasses import dataclass
from typing import List, Tuple

# ==================== CONFIGURATION ====================

@dataclass
class DroneConfig:
    """Drone physical parameters"""
    max_speed: float = 3.0
    max_acceleration: float = 2.0
    drag_coefficient: float = 0.3
    inertia_factor: float = 0.85
    turn_rate_limit: float = 5.0
    mass: float = 1.0

@dataclass
class EnvironmentConfig:
    """Environment parameters"""
    map_size: Tuple[float, float] = (10.0, 10.0)
    num_obstacles: int = 5
    obstacle_size_range: Tuple[float, float] = (0.5, 1.5)
    num_goals: int = 5
    lidar_rays: int = 8
    lidar_range: float = 3.0
    goal_reach_threshold: float = 0.3
    collision_threshold: float = 0.2
    
@dataclass
class TrainingConfig:
    """PPO training parameters"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    
@dataclass
class CurriculumConfig:
    """Curriculum learning stages"""
    stage: int = 0
    max_goals: int = 5
    stages: List[dict] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = [
                {"num_obstacles": 2, "max_speed": 2.0, "num_goals": 3},
                {"num_obstacles": 4, "max_speed": 2.5, "num_goals": 4},
                {"num_obstacles": 6, "max_speed": 3.0, "num_goals": 5},
            ]

# ==================== ENVIRONMENT ====================

class DroneNavigationEnv(gym.Env):
    """2D Continuous Drone Navigation Environment with Lidar"""
    
    def __init__(self, drone_config: DroneConfig, env_config: EnvironmentConfig, max_goals: int = 5):
        super().__init__()
        self.drone_config = drone_config
        self.env_config = env_config
        self.max_goals = max_goals
        
        state_dim = 8 + self.max_goals + env_config.lidar_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.drone_pos = np.zeros(2)
        self.drone_vel = np.zeros(2)
        self.goals = []
        self.goals_visited = []
        self.obstacles = []
        self.current_goal_idx = 0
        self.steps = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.drone_pos = np.array([
            np.random.uniform(1, self.env_config.map_size[0] - 1),
            np.random.uniform(1, self.env_config.map_size[1] - 1)
        ])
        self.drone_vel = np.zeros(2)
        
        self.obstacles = self._generate_obstacles()
        self.goals = self._generate_goals()
        self.goals_visited = [False] * self.env_config.num_goals
        self.current_goal_idx = 0
        self.steps = 0
        
        return self._get_observation(), {}
    
    def _generate_obstacles(self) -> List[Tuple[np.ndarray, float]]:
        """Generate random obstacles (position, radius)"""
        obstacles = []
        for _ in range(self.env_config.num_obstacles):
            valid = False
            while not valid:
                pos = np.array([
                    np.random.uniform(0, self.env_config.map_size[0]),
                    np.random.uniform(0, self.env_config.map_size[1])
                ])
                radius = np.random.uniform(*self.env_config.obstacle_size_range)
                
                if np.linalg.norm(pos - self.drone_pos) > radius + 1.0:
                    valid = True
            obstacles.append((pos, radius))
        return obstacles
    
    def _generate_goals(self) -> List[np.ndarray]:
        """Generate random goal positions"""
        goals = []
        for _ in range(self.env_config.num_goals):
            valid = False
            while not valid:
                pos = np.array([
                    np.random.uniform(0.5, self.env_config.map_size[0] - 0.5),
                    np.random.uniform(0.5, self.env_config.map_size[1] - 0.5)
                ])
                
                valid = True
                for obs_pos, obs_radius in self.obstacles:
                    if np.linalg.norm(pos - obs_pos) < obs_radius + 0.5:
                        valid = False
                        break
            goals.append(pos)
        return goals
    
    def _get_lidar_readings(self) -> np.ndarray:
        """Simulate lidar sensor readings"""
        readings = np.ones(self.env_config.lidar_rays) * self.env_config.lidar_range
        angles = np.linspace(0, 2 * np.pi, self.env_config.lidar_rays, endpoint=False)
        
        for i, angle in enumerate(angles):
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            min_dist = self.env_config.lidar_range
            for obs_pos, obs_radius in self.obstacles:
                to_obstacle = obs_pos - self.drone_pos
                proj_length = np.dot(to_obstacle, direction)
                
                if proj_length > 0:
                    closest_point = self.drone_pos + direction * proj_length
                    dist_to_center = np.linalg.norm(closest_point - obs_pos)
                    
                    if dist_to_center < obs_radius:
                        dist = proj_length - np.sqrt(obs_radius**2 - dist_to_center**2)
                        min_dist = min(min_dist, max(0, dist))
            
            for axis in [0, 1]:
                if abs(direction[axis]) > 0.01:
                    for boundary in [0, self.env_config.map_size[axis]]:
                        t = (boundary - self.drone_pos[axis]) / direction[axis]
                        if t > 0:
                            min_dist = min(min_dist, t)
            
            readings[i] = min_dist / self.env_config.lidar_range
        
        return readings
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation with fixed-size goal vector"""
        unvisited_goals = [i for i, visited in enumerate(self.goals_visited) if not visited]
        if unvisited_goals:
            distances = [np.linalg.norm(self.goals[i] - self.drone_pos) for i in unvisited_goals]
            self.current_goal_idx = unvisited_goals[np.argmin(distances)]
        
        current_goal = self.goals[self.current_goal_idx]
        to_goal = current_goal - self.drone_pos
        goal_dist = np.linalg.norm(to_goal)
        goal_angle = np.arctan2(to_goal[1], to_goal[0])
        
        norm_pos = self.drone_pos / self.env_config.map_size[0]
        norm_vel = self.drone_vel / self.drone_config.max_speed
        norm_goal = to_goal / self.env_config.map_size[0]
        
        lidar = self._get_lidar_readings()
        
        goals_visited_padded = self.goals_visited + [False] * (self.max_goals - len(self.goals_visited))
        
        obs = np.concatenate([
            norm_pos,
            norm_vel,
            norm_goal,
            [goal_dist / self.env_config.map_size[0]],
            [goal_angle / np.pi],
            goals_visited_padded[:self.max_goals],
            lidar
        ]).astype(np.float32)
        
        return obs
    
    def _check_collision(self) -> bool:
        """Check if drone collides with obstacles or boundaries"""
        for obs_pos, obs_radius in self.obstacles:
            if np.linalg.norm(self.drone_pos - obs_pos) < obs_radius + self.env_config.collision_threshold:
                return True
        
        if (self.drone_pos[0] < 0 or self.drone_pos[0] > self.env_config.map_size[0] or
            self.drone_pos[1] < 0 or self.drone_pos[1] > self.env_config.map_size[1]):
            return True
        
        return False
    
    def step(self, action: np.ndarray):
        self.steps += 1
        
        acceleration = action * self.drone_config.max_acceleration
        
        self.drone_vel = (self.drone_vel * self.drone_config.inertia_factor + 
                         acceleration * (1.0 / self.drone_config.mass))
        
        speed = np.linalg.norm(self.drone_vel)
        if speed > 0:
            drag = -self.drone_config.drag_coefficient * self.drone_vel * speed
            self.drone_vel += drag
        
        speed = np.linalg.norm(self.drone_vel)
        if speed > self.drone_config.max_speed:
            self.drone_vel = self.drone_vel / speed * self.drone_config.max_speed
        
        self.drone_pos += self.drone_vel * 0.1
        
        reward = 0.0
        terminated = False
        
        if self._check_collision():
            reward = -100.0
            terminated = True
        else:
            current_goal = self.goals[self.current_goal_idx]
            dist_to_goal = np.linalg.norm(current_goal - self.drone_pos)
            
            if dist_to_goal < self.env_config.goal_reach_threshold:
                if not self.goals_visited[self.current_goal_idx]:
                    reward = 50.0
                    self.goals_visited[self.current_goal_idx] = True
                    
                    if all(self.goals_visited):
                        reward = 100.0
                        terminated = True
            
            reward += -0.1
            reward += -dist_to_goal * 0.01
            
            if speed > 0.1:
                direction_to_goal = (current_goal - self.drone_pos) / (dist_to_goal + 1e-6)
                velocity_direction = self.drone_vel / (speed + 1e-6)
                alignment = np.dot(direction_to_goal, velocity_direction)
                reward += alignment * 0.1
        
        if self.steps >= self.max_steps:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, {}

# ==================== PPO AGENT ====================

class ActorCritic(nn.Module):
    """Actor-Critic neural network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.feature(state)
        return features
    
    def act(self, state):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action, action_logprob
    
    def evaluate(self, state, action):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return action_logprob, value, entropy

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: TrainingConfig):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'logprobs': [],
            'values': []
        }
    
    def select_action(self, state: np.ndarray):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, logprob = self.policy.act(state_tensor)
            value = self.policy.critic(self.policy.forward(state_tensor)).squeeze()
        
        return action.cpu().numpy()[0], logprob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, action, reward, done, logprob, value):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['logprobs'].append(logprob)
        self.buffer['values'].append(value)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        values = np.array(self.buffer['values'] + [next_value])
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update(self, next_state):
        """PPO update step"""
        if len(self.buffer['states']) < self.config.batch_size:
            return {}
        
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.policy.critic(
                self.policy.forward(next_state_tensor)
            ).cpu().item()
        
        advantages, returns = self.compute_gae(next_value)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer['logprobs'])).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
        for _ in range(self.config.ppo_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                logprobs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                ratio = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                   1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.mean().item())
        
        for key in self.buffer:
            self.buffer[key] = []
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ==================== TRAINING SYSTEM ====================

class CurriculumTrainer:
    """Training system with curriculum learning"""
    
    def __init__(self, 
                 drone_config: DroneConfig,
                 env_config: EnvironmentConfig,
                 training_config: TrainingConfig,
                 curriculum_config: CurriculumConfig):
        self.drone_config = drone_config
        self.env_config = env_config
        self.training_config = training_config
        self.curriculum_config = curriculum_config
        
        self.env = DroneNavigationEnv(drone_config, env_config, max_goals=curriculum_config.max_goals)
        self.agent = PPOAgent(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            training_config
        )
        
        self.metrics = {
            'episode_rewards': [],
            'success_rate': [],
            'avg_steps': [],
            'curriculum_stage': []
        }
    
    def apply_curriculum_stage(self, stage: int):
        """Apply curriculum stage settings"""
        if stage < len(self.curriculum_config.stages):
            stage_config = self.curriculum_config.stages[stage]
            self.env_config.num_obstacles = stage_config['num_obstacles']
            self.drone_config.max_speed = stage_config['max_speed']
            self.env_config.num_goals = stage_config['num_goals']
            self.env = DroneNavigationEnv(self.drone_config, self.env_config, 
                                         max_goals=self.curriculum_config.max_goals)
            print(f"\n🎯 Curriculum Stage {stage + 1}: {stage_config}")
    
    def should_advance_curriculum(self, recent_success_rate: float) -> bool:
        """Determine if ready to advance to next stage"""
        return recent_success_rate > 0.8
    
    def train(self, total_episodes: int = 1000, eval_interval: int = 10):
        """Main training loop"""
        print("🚁 Starting Drone Navigation Training with PPO + Curriculum Learning")
        print(f"Device: {self.agent.device}")
        
        current_stage = 0
        self.apply_curriculum_stage(current_stage)
        
        episode = 0
        recent_successes = []
        
        while episode < total_episodes:
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                action, logprob, value = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.agent.store_transition(state, action, reward, done, logprob, value)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if len(self.agent.buffer['states']) >= self.training_config.buffer_size:
                    self.agent.update(next_state)
            
            success = all(self.env.goals_visited)
            recent_successes.append(1.0 if success else 0.0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['success_rate'].append(np.mean(recent_successes))
            self.metrics['avg_steps'].append(steps)
            self.metrics['curriculum_stage'].append(current_stage)
            
            episode += 1
            
            if episode % eval_interval == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-eval_interval:])
                success_rate = np.mean(recent_successes)
                print(f"Episode {episode} | Stage {current_stage + 1} | "
                      f"Reward: {avg_reward:.2f} | Success: {success_rate:.2%} | "
                      f"Steps: {steps}")
            
            if episode % 100 == 0 and len(recent_successes) >= 50:
                recent_sr = np.mean(recent_successes[-50:])
                if self.should_advance_curriculum(recent_sr):
                    if current_stage < len(self.curriculum_config.stages) - 1:
                        current_stage += 1
                        self.apply_curriculum_stage(current_stage)
                        recent_successes = []
        
        print("\n✅ Training Complete!")
        return self.metrics
    
    def train_with_visualization(self, total_episodes: int = 1000, render_every: int = 100):
        """Training with periodic visualization"""
        print("🚁 Starting Training with Visualization")
        print(f"Device: {self.agent.device}")
        
        current_stage = 0
        self.apply_curriculum_stage(current_stage)
        
        episode = 0
        recent_successes = []
        
        while episode < total_episodes:
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            trajectory = [self.env.drone_pos.copy()]
            
            while not done:
                action, logprob, value = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.agent.store_transition(state, action, reward, done, logprob, value)
                
                state = next_state
                episode_reward += reward
                steps += 1
                trajectory.append(self.env.drone_pos.copy())
                
                if len(self.agent.buffer['states']) >= self.training_config.buffer_size:
                    self.agent.update(next_state)
            
            success = all(self.env.goals_visited)
            recent_successes.append(1.0 if success else 0.0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['success_rate'].append(np.mean(recent_successes))
            self.metrics['avg_steps'].append(steps)
            self.metrics['curriculum_stage'].append(current_stage)
            
            episode += 1
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
                success_rate = np.mean(recent_successes)
                print(f"Episode {episode} | Stage {current_stage + 1} | "
                      f"Reward: {avg_reward:.2f} | Success: {success_rate:.2%} | Steps: {steps}")
            
            if episode % render_every == 0:
                print(f"📊 Visualizing Episode {episode}...")
                self._quick_render(trajectory)
            
            if episode % 100 == 0 and len(recent_successes) >= 50:
                recent_sr = np.mean(recent_successes[-50:])
                if self.should_advance_curriculum(recent_sr):
                    if current_stage < len(self.curriculum_config.stages) - 1:
                        current_stage += 1
                        self.apply_curriculum_stage(current_stage)
                        recent_successes = []
        
        print("\n✅ Training Complete!")
        return self.metrics
    
    def _quick_render(self, trajectory):
        """Quick static visualization during training"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for obs_pos, obs_radius in self.env.obstacles:
            circle = Circle(obs_pos, obs_radius, color='gray', alpha=0.5)
            ax.add_patch(circle)
        
        for i, goal in enumerate(self.env.goals):
            color = 'green' if self.env.goals_visited[i] else 'red'
            circle = Circle(goal, 0.2, color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(goal[0], goal[1], f'G{i+1}', ha='center', va='center')
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.6)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        
        ax.set_xlim(0, self.env_config.map_size[0])
        ax.set_ylim(0, self.env_config.map_size[1])
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Training Progress Snapshot')
        ax.grid(True, alpha=0.3)
        plt.pause(0.5)
        plt.close()
    
    def animate_episode(self, num_episodes=1, interval=50):
        """Watch the drone navigate in real-time animation"""
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            for obs_pos, obs_radius in self.env.obstacles:
                circle = Circle(obs_pos, obs_radius, color='gray', alpha=0.5)
                ax.add_patch(circle)
            
            goal_patches = []
            for i, goal in enumerate(self.env.goals):
                circle = Circle(goal, 0.2, color='red', alpha=0.7)
                ax.add_patch(circle)
                goal_patches.append(circle)
                ax.text(goal[0], goal[1], f'G{i+1}', ha='center', va='center')
            
            drone_marker, = ax.plot([], [], 'bo', markersize=15, label='Drone')
            trail_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.4, label='Path')
            
            lidar_lines = []
            for _ in range(self.env_config.lidar_rays):
                line, = ax.plot([], [], 'r-', linewidth=1, alpha=0.3)
                lidar_lines.append(line)
            
            ax.set_xlim(0, self.env_config.map_size[0])
            ax.set_ylim(0, self.env_config.map_size[1])
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title(f'Live Drone Navigation - Episode {ep + 1}')
            ax.grid(True, alpha=0.3)
            
            trajectory = [self.env.drone_pos.copy()]
            
            def update(frame):
                nonlocal state, done
                
                if not done:
                    action, _, _ = self.agent.select_action(state)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    trajectory.append(self.env.drone_pos.copy())
                    
                    drone_marker.set_data([self.env.drone_pos[0]], [self.env.drone_pos[1]])
                    
                    traj_array = np.array(trajectory)
                    trail_line.set_data(traj_array[:, 0], traj_array[:, 1])
                    
                    for i, visited in enumerate(self.env.goals_visited):
                        if visited:
                            goal_patches[i].set_color('green')
                    
                    angles = np.linspace(0, 2 * np.pi, self.env_config.lidar_rays, endpoint=False)
                    lidar_readings = self.env._get_lidar_readings()
                    
                    for i, (angle, reading) in enumerate(zip(angles, lidar_readings)):
                        distance = reading * self.env_config.lidar_range
                        end_x = self.env.drone_pos[0] + distance * np.cos(angle)
                        end_y = self.env.drone_pos[1] + distance * np.sin(angle)
                        lidar_lines[i].set_data(
                            [self.env.drone_pos[0], end_x],
                            [self.env.drone_pos[1], end_y]
                        )
                
                return [drone_marker, trail_line] + lidar_lines + goal_patches
            
            anim = FuncAnimation(fig, update, frames=1000, interval=interval, 
                               blit=True, repeat=False)
            
            plt.show()
            
            print(f"Episode {ep + 1}: Goals reached: {sum(self.env.goals_visited)}/{len(self.env.goals)}")
    
    def evaluate(self, num_episodes: int = 10, render: bool = True):
        """Evaluate trained agent"""
        print(f"\n📊 Evaluating agent for {num_episodes} episodes...")
        
        successes = 0
        total_rewards = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            trajectory = [self.env.drone_pos.copy()]
            
            while not done:
                action, _, _ = self.agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                trajectory.append(self.env.drone_pos.copy())
            
            if all(self.env.goals_visited):
                successes += 1
            
            total_rewards.append(episode_reward)
            
            if render and ep == 0:
                self._render_episode(trajectory)
        
        print(f"Success Rate: {successes / num_episodes:.2%}")
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
    
    def _render_episode(self, trajectory: List[np.ndarray]):
        """Render a single episode"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for obs_pos, obs_radius in self.env.obstacles:
            circle = Circle(obs_pos, obs_radius, color='gray', alpha=0.5)
            ax.add_patch(circle)
        
        for i, goal in enumerate(self.env.goals):
            color = 'green' if self.env.goals_visited[i] else 'red'
            circle = Circle(goal, 0.2, color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(goal[0], goal[1], f'G{i+1}', ha='center', va='center')
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.6)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'bs', markersize=10, label='End')
        
        ax.set_xlim(0, self.env_config.map_size[0])
        ax.set_ylim(0, self.env_config.map_size[1])
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Drone Navigation Trajectory')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_training_metrics(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(self.metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        axes[0, 1].plot(self.metrics['success_rate'])
        axes[0, 1].set_title('Success Rate (Rolling Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        
        axes[1, 0].plot(self.metrics['avg_steps'])
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        axes[1, 1].plot(self.metrics['curriculum_stage'])
        axes[1, 1].set_title('Curriculum Stage')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Stage')
        
        plt.tight_layout()
        plt.show()

# ==================== INTERACTIVE CONFIGURATION ====================

def create_interactive_trainer():
    """Create trainer with interactive configuration"""
    
    print("="*60)
    print("🚁 AUTONOMOUS DRONE NAVIGATION - PPO + CURRICULUM LEARNING")
    print("="*60)
    
    print("\n⚙️  CONFIGURATION OPTIONS\n")
    
    print("📐 Drone Physics:")
    max_speed = float(input("  Max Speed (default 3.0): ") or "3.0")
    inertia = float(input("  Inertia Factor 0-1 (default 0.85, higher=more drift): ") or "0.85")
    drag = float(input("  Drag Coefficient (default 0.3): ") or "0.3")
    
    print("\n🗺️  Environment:")
    num_obstacles = int(input("  Number of Obstacles (default 5): ") or "5")
    obstacle_size = float(input("  Obstacle Size Range Max (default 1.5): ") or "1.5")
    num_goals = int(input("  Number of Goals (default 5): ") or "5")
    
    print("\n🧠 Training:")
    total_episodes = int(input("  Total Episodes (default 1000): ") or "1000")
    
    drone_config = DroneConfig(
        max_speed=max_speed,
        inertia_factor=inertia,
        drag_coefficient=drag
    )
    
    env_config = EnvironmentConfig(
        num_obstacles=num_obstacles,
        obstacle_size_range=(0.5, obstacle_size),
        num_goals=num_goals
    )
    
    training_config = TrainingConfig()
    curriculum_config = CurriculumConfig(max_goals=num_goals)
    
    print("\n✨ Configuration Complete! Starting training...\n")
    
    trainer = CurriculumTrainer(
        drone_config, env_config, training_config, curriculum_config
    )
    
    return trainer, total_episodes

# ==================== MAIN ====================

if __name__ == "__main__":
    use_interactive = input("Use interactive configuration? (y/n, default y): ").lower() != 'n'
    
    if use_interactive:
        trainer, total_episodes = create_interactive_trainer()
    else:
        drone_config = DroneConfig()
        env_config = EnvironmentConfig()
        training_config = TrainingConfig()
        curriculum_config = CurriculumConfig(max_goals=5)
        
        trainer = CurriculumTrainer(
            drone_config, env_config, training_config, curriculum_config
        )
        total_episodes = 1000
    
    # Choose training mode
    use_visualization = input("Show training visualization? (y/n, default n): ").lower() == 'y'
    
    if use_visualization:
        metrics = trainer.train_with_visualization(
            total_episodes=total_episodes, 
            render_every=100
        )
    else:
        metrics = trainer.train(total_episodes=total_episodes, eval_interval=10)
    
    # Save model to home directory
    save_path = os.path.expanduser("~/drone_navigation_ppo.pth")
    trainer.agent.save(save_path)
    print(f"\n💾 Model saved to {save_path}")
    
    trainer.plot_training_metrics()
    
    trainer.evaluate(num_episodes=5, render=True)
