import sys
sys.path.append("./")

from configparser import ConfigParser, ExtendedInterpolation

from src.environment import create_kicker_env
from src.sb3.stable_baselines3 import A2C, DQN, PPO

from src.sb3_contrib.sb3_contrib import TQC
from src.tensorboard_aggregator import aggregator
from train import train_kicker
from src.evaluate import evaluate_model

from pathlib import Path


def main():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('./resources/config.ini')
    used_rl_algorithm = PPO
    execution_mode = config['Common']['mode']
    if execution_mode == 'train':
        for seed in range(1, 2):
            env = create_kicker_env(config=config, seed=seed)
            train_kicker(config=config, seed=seed, algorithm_class=used_rl_algorithm, env=env)
        aggregator.main(path_arg=config['Algorithm']['tensorboard_log'])
    elif execution_mode == 'test':
        save_path = Path(__file__).parent.parent / "testing"
        save_path.exists() or save_path.mkdir(parents=True, exist_ok=True)
        evaluate_model(config=config, save_path=save_path, algorithm_class=used_rl_algorithm)  # Set normalize_env_path (dateiende .pkl) if normalized wrapper was used during training
    else:
        raise ValueError(f"Unknown mode: {execution_mode}")

if __name__ == '__main__':
    main()
