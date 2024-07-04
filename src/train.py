import ast
from configparser import ConfigParser
from src.config_logging import save_run_info
from src.sb3.stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, TensorboardCallback


def train_kicker(config: ConfigParser, seed: int, algorithm_class, env):
    alg_config = config['Algorithm']
    dgn_config = config['DQN']
    ppo_config = config['PPO']
    try:
        policy_kwargs = ast.literal_eval(alg_config['policy_kwargs'])
        #This code creates our trainer, currently using the A2C algorithm
        model = algorithm_class(env=env, seed=seed, verbose=1,
                                policy=alg_config['policy'],
                                policy_kwargs=policy_kwargs,
                                tensorboard_log=alg_config['tensorboard_log'],

                                #hyperparameters from DQN-Section
                                #learning_rate=dgn_config.getfloat('learning_rate'),
                                #buffer_size=dgn_config.getint('buffer_size'),
                                #batch_size=dgn_config.getint('batch_size'),
                                #gamma=dgn_config.getfloat('gamma'),
                                #exploration_fraction=dgn_config.getfloat('exploration_fraction'),
                                
                                #hyperparameters from PPO-Section
                                #clip_range_vf = ppo_config.getfloat('clip_range_vf'),
                                #max_grad_norm = ppo_config.getfloat('max_grad_norm'),
                                #ent_coef = ppo_config.getfloat('ent_coef'),
                                #vf_coef = ppo_config.getfloat('vf_coef'),   
                                #normalize_advantage = ppo_config.getboolean('normalize_advantage'),
                                #learning_rate = ppo_config.getfloat('learning_rate'),
                                #n_steps = ppo_config.getint('n_steps'),
                                #batch_size = ppo_config.getint('batch_size'),
                                #n_epochs = ppo_config.getint('n_epochs'),
                                gamma = ppo_config.getfloat('gamma'),
                                #gae_lambda = ppo_config.getfloat('gae_lambda'),
                                #clip_range = ppo_config.getfloat('clip_range'), 

                                ################################
                                # Add here more hyperparameters if needed, following the above scheme
                                # alg_config['hyperparameter_name']
                                ################################
                                )
    except KeyError or ValueError:
        # Fall back to default policy_kwargs
        print("No policy_kwargs found in config, using default policy_kwargs")
        model = algorithm_class(env=env, seed=seed, verbose=1,
                                policy=alg_config['policy'],
                                tensorboard_log=alg_config['tensorboard_log'],
                                )

    save_run_info(config=config,
                  seed=seed,
                  algorithm_name=type(model).__name__)

    training_config = config['Training']
    model.learn(total_timesteps=int(training_config['total_timesteps']),
                tb_log_name=training_config['tb_log_name'],
                callback=get_callback(config, seed))
    env.close()


def get_callback(config: ConfigParser, seed: int):
    callback_config = config['Callback']
    checkpoint_callback = CheckpointCallback(name_prefix=f"rl_model_{seed}",
                              save_freq=int(callback_config['save_freq']),
                              save_path=callback_config['save_path'],
                              save_replay_buffer=callback_config.getboolean('save_replay_buffer'),
                              save_vecnormalize=callback_config.getboolean('save_vecnormalize'))
    logging_callback = TensorboardCallback()
    return CallbackList([checkpoint_callback, logging_callback])
