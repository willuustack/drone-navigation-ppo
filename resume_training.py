import sys
import os
from drone_lib import *

# Accept model name from command line (default remains the balanced file)
model_name = sys.argv[1] if len(sys.argv) > 1 else "drone_navigation_balanced.pth"
model_path = os.path.join("models", model_name)

# Build trainer
drone_config = DroneConfig()
env_config = EnvironmentConfig()
training_config = TrainingConfig()
curriculum_config = CurriculumConfig()
trainer = CurriculumTrainer(drone_config, env_config, training_config, curriculum_config)

# Load checkpoint
if os.path.exists(model_path):
    print(f"✅ Found saved model at {model_path}. Resuming training...")
    trainer.agent.load(model_path)
else:
    print(f"❌ ERROR: Model not found at {model_path}. Cannot resume training.")
    print("Please run 'python drone_nav.py' to perform the initial training first.")
    sys.exit(1)

# Continue training
additional_episodes = 50000
print(f"\n🚀 Continuing training for an additional {additional_episodes} episodes...")
show_viz = input("Show occasional training visualizations? (y/n, default n): ").lower() == 'y'
if show_viz:
    trainer.train_with_visualization(total_episodes=additional_episodes, render_every=100)
else:
    trainer.train(total_episodes=additional_episodes)

# Save updated policy (ask user for a filename)
save_name = input("Enter filename to save improved model (default: drone_navigation_balanced.pth): ").strip()
if not save_name:
    save_name = "drone_navigation_balanced.pth"
trainer.save_model(save_name)

print("\n📊 Evaluating the newly improved model...")
trainer.evaluate(num_episodes=10)
