import os
from drone_lib import DroneConfig, EnvironmentConfig, TrainingConfig, CurriculumConfig, CurriculumTrainer

def select_model(models_dir="models"):
    files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not files:
        print(f"No model files found in '{models_dir}'")
        exit(1)

    print("Available models:")
    for i, fname in enumerate(files, start=1):
        print(f"  {i}. {fname}")

    while True:
        choice = input(f"Enter model number (1{len(files)}): ").strip()
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

    print(f"\n✅ Found saved model at {model_path}. Resuming training...")
    trainer.agent.load(model_path)

    additional_episodes = 100000 #SET THIS TO WHATEVER YOU WANT
    print(f"\n🚀 Continuing training for an additional {additional_episodes} episodes...")
    show_viz = input("Show occasional training visualizations? (y/n, default n): ").lower() == 'y'
    if show_viz:
        trainer.train_with_visualization(total_episodes=additional_episodes, render_every=100)
    else:
        trainer.train(total_episodes=additional_episodes)

    save_name = input("Enter filename to save improved model (default: drone_navigation_balanced.pth): ").strip()
    if not save_name:
        save_name = "drone_navigation_balanced.pth"
    trainer.save_model(save_name)

    print("\n📊 Evaluating the newly improved model...")
    trainer.evaluate(num_episodes=10)
