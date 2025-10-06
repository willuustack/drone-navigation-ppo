# Change this line in watch_drone.py
from drone_lib import DroneConfig, EnvironmentConfig, TrainingConfig, CurriculumConfig, CurriculumTrainer
import os


# --- Create the trainer instance ---
# These classes are imported from drone_lib.py
drone_config = DroneConfig()
env_config = EnvironmentConfig()
training_config = TrainingConfig()
curriculum_config = CurriculumConfig()
trainer = CurriculumTrainer(drone_config, env_config, training_config, curriculum_config)


# --- Define the model path ---
model_path = "models/drone_navigation_balanced.pth"


# --- Load the trained model ---
if os.path.exists(model_path):
    print(f"✅ Loading balanced model from {model_path}")
    trainer.agent.load(model_path)
    
    # --- Watch the drone in action ---
    print("\n🎬 Starting balanced navigation animation!")
    # This will now call the fully implemented animation method
    trainer.animate_episode(num_episodes=10, interval=40) 
else:
    print(f"❌ Model not found at {model_path}.")
    print("Please run 'python3 drone_nav.py' to train the model first.")

