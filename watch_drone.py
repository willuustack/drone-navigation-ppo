import os
from drone_lib import DroneConfig, EnvironmentConfig, TrainingConfig, CurriculumConfig, CurriculumTrainer

def select_model(models_dir="models"):
    # List all .pth files in the models directory
    files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not files:
        print(f"No model files found in '{models_dir}'")
        exit(1)

    print("Available models:")
    for i, fname in enumerate(files, start=1):
        print(f"  {i}. {fname}")

    # Prompt until valid selection
    while True:
        choice = input(f"Enter model number (1–{len(files)}): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return os.path.join(models_dir, files[idx - 1])
        print("Invalid choice, please try again.")

if __name__ == "__main__":
    model_path = select_model()

    drone_config = DroneConfig()
    env_config = EnvironmentConfig()
    training_config = TrainingConfig()
    curriculum_config = CurriculumConfig()
    trainer = CurriculumTrainer(drone_config, env_config, training_config, curriculum_config)

    print(f"\n✅ Loading model from {model_path}")
    trainer.agent.load(model_path)

    print("\n🎬 Starting navigation animation!")
    trainer.animate_episode(num_episodes=10, interval=40)
