from drone_nav import *
import os
import matplotlib
matplotlib.use('MacOSX')

# Load trained model
drone_config = DroneConfig()
env_config = EnvironmentConfig()
training_config = TrainingConfig()
curriculum_config = CurriculumConfig(max_goals=5)

trainer = CurriculumTrainer(
    drone_config, env_config, training_config, curriculum_config
)

# Load saved model
save_path = os.path.expanduser("~/Desktop/drone_nav_project/drone_navigation_ppo.pth")
trainer.agent.load(save_path)
print("✅ Model loaded successfully!")

# Watch the drone in action with live animation
print("\n🎬 Starting live animation...")
trainer.animate_episode(num_episodes=3, interval=30)
