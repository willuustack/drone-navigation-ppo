# In resume_training.py

from drone_lib import *
import os
import sys

# --- Configuration (must match the "Balanced" settings) ---
drone_config = DroneConfig()
env_config = EnvironmentConfig()
training_config = TrainingConfig()
curriculum_config = CurriculumConfig()
trainer = CurriculumTrainer(drone_config, env_config, training_config, curriculum_config)

# --- Define the model path ---
model_path = "models/drone_navigation_balanced.pth"

# --- Load the saved model ---
if os.path.exists(model_path):
    print(f"✅ Found saved model at {model_path}. Resuming training...")
    trainer.agent.load(model_path)
else:
    # This is a safety check. If the model doesn't exist, we can't resume.
    print(f"❌ ERROR: Model not found at {model_path}. Cannot resume training.")
    print("Please run 'python3 drone_nav.py' to perform the initial training first.")
    sys.exit() # Exit the script if there's nothing to resume.

# --- Continue Training ---
# Let's train for another 750 episodes to master Stage 2 & 3
additional_episodes = 50000
print(f"\n🚀 Continuing training for an additional {additional_episodes} episodes...")

# Ask the user if they want to see visualizations
show_viz = input("Show occasional training visualizations? (y/n, default n): ").lower() == 'y'

if show_viz:
    # This method now exists in your updated drone_nav.py
    trainer.train_with_visualization(total_episodes=additional_episodes, render_every=100)
else:
    # The standard, faster training method
    trainer.train(total_episodes=additional_episodes)

# --- Save the improved model ---
# This saves the newly refined model back to the same file
trainer.save_model()

# --- Evaluate the final, improved model ---
print("\n📊 Evaluating the newly improved model...")
trainer.evaluate(num_episodes=10)
