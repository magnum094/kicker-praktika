import time
from configparser import ConfigParser
from pathlib import Path

from environment import load_normalized_kicker_env, create_kicker_env
from sb3.stable_baselines3.common.evaluation import evaluate_policy


def evaluate_model(config: ConfigParser, save_path: Path, algorithm_class, normalize_env_path: str = None):
    if normalize_env_path is None:
        env = create_kicker_env(seed=config['Testing'].getint('eval_seed'), config=config)
    else:
        env = load_normalized_kicker_env(config=config, seed=config['Testing'].getint('eval_seed'),
                                         normalize_path=config['Testing']['normalized_env_path'])
    model = algorithm_class.load(config['Testing']['test_model_path'], env=env)
    episode_rewards, episode_lengths = evaluate_policy(model=model, env=env,
                                                       n_eval_episodes=config['Testing'].getint('num_eval_episodes'))
    save_results(config=config, save_path=save_path,
                 episode_rewards=episode_rewards, episode_lengths=episode_lengths)

    print("-" * 50)
    print(f"Mean reward: {episode_rewards}, Mean episode length: {episode_lengths}")
    print("-" * 50)


def save_results(config: ConfigParser, save_path: Path, episode_rewards, episode_lengths):
    with open(save_path / f'evaluation_result_{round(time.time() * 1000)}.txt', 'w') as f:
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Seed: {config['Testing'].getint('eval_seed')}\n")
        f.write(f"Number of evaluation episodes: {config['Testing'].getint('num_eval_episodes')}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean reward: {episode_rewards}\n")
        f.write(f"Mean episode length: {episode_lengths}\n")
        f.write("-" * 50 + "\n")
