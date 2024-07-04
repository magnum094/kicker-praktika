from configparser import ConfigParser

from numpy import ndarray

from kicker.kicker_env import Kicker
from src.sb3.stable_baselines3.common.monitor import Monitor
from src.sb3.stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecEnvWrapper, VecEnv
from src.sb3.stable_baselines3.common.vec_env.vec_pbrs import VecPBRSWrapper
from src.sb3.stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from gymnasium.wrappers import TransformReward
from gymnasium import Wrapper
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from src import custom_wrappers



def create_kicker_env(config: ConfigParser, seed: int):
    env_conf = config['Kicker']
    env = Kicker(seed=seed,
                 horizon=int(env_conf['horizon']),
                 continuous_act_space=env_conf.getboolean('continuous_act_space'),
                 multi_discrete_act_space=env_conf.getboolean('multi_discrete_act_space'),
                 image_obs_space=env_conf.getboolean('image_obs_space'),
                 end_episode_on_struck_goal=env_conf.getboolean('end_episode_on_struck_goal'),
                 end_episode_on_conceded_goal=env_conf.getboolean('end_episode_on_conceded_goal'),
                 reset_goalie_position=env_conf.getboolean('reset_goalie_position'),
                 render_training=env_conf.getboolean('render_training'),
                 lateral_bins=env_conf.getint('lateral_bins'),
                 angular_bins=env_conf.getint('angular_bins'),
                 step_frequency=env_conf.getint('step_frequency'))
    
    
   
       

    # Default wrappers
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    ############################################
    # Add Wrappers here
    ############################################
    #env = custom_wrappers.VecScaleNegReward(env, scale_factor=0)
    #env = custom_wrappers.VecScalePositivReward(env, scale_factor=10)
    kicker_config = config['Kicker']
    #env = custom_wrappers.VecLingeringRewardTime(env, total_timesteps=kicker_config.getint('horizon'))
    #env = custom_wrappers.VecLingeringRewardDistance(env, goal_post=[1.296, 0, 0.2], total_timesteps=kicker_config.getint('horizon'))

    ppo_config = config['PPO']
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=ppo_config.getfloat('gamma'))


    
    if not env_conf.getboolean('render_training'):
        video_conf = config['VideoRecording']
        print(f"Recording video every {video_conf.getint('video_interval')} steps with a length of "
              f"{video_conf.getint('video_length')} frames, saving to {video_conf['video_folder']}")
        env = VecVideoRecorder(venv=env, name_prefix=f"rl-kicker-video-{seed}",
                               record_video_trigger=lambda x: x % video_conf.getint('video_interval') == 0,
                               video_length=video_conf.getint('video_length'),
                               video_folder=video_conf['video_folder'])
    env.seed(seed)
    return env


def load_normalized_kicker_env(config: ConfigParser, seed: int, normalize_path: str):
    env = create_kicker_env(seed=seed, config=config)
    env = VecNormalize.load(normalize_path, env)
    return env

