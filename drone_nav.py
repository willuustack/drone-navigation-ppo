from drone_lib import DroneConfig, EnvironmentConfig, TrainingConfig, CurriculumConfig, CurriculumTrainer
import os

if __name__ == "__main__":
    total_episodes = 1500
    
    drone_config = DroneConfig()
    env_config = EnvironmentConfig()
    training_config = TrainingConfig()
    curriculum_config = CurriculumConfig(max_goals=5)
    trainer = CurriculumTrainer(drone_config, env_config, training_config, curriculum_config)
    
    show_viz = input("Show occasional training visualizations? (y/n, default n): ").lower() == 'y'

    if show_viz:
        trainer.train_with_visualization(total_episodes=total_episodes, render_every=100)
    else:
        trainer.train(total_episodes=total_episodes)
    
    trainer.save_model()
    trainer.evaluate(num_episodes=10)
