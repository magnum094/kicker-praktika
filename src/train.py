import ast
from configparser import ConfigParser
from src.config_logging import save_run_info
from src.sb3.stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, TensorboardCallback


def train_kicker(config: ConfigParser, seed: int, algorithm_class, env):
    alg_config = config['Algorithm']
    try:
        policy_kwargs = ast.literal_eval(alg_config['policy_kwargs'])
        #This code creates our trainer, currently using the A2C algorithm
        model = algorithm_class(env=env, seed=seed, verbose=1,
                                policy=alg_config['policy'],
                                policy_kwargs=policy_kwargs,
                                #tensorboard_log=alg_config['tensorboard_log'],
                                #learning_rate=alg_config.getfloat('learning_rate'),
                                #buffer_size=alg_config.getint('buffer_size'),
                                #batch_size=alg_config.getint('batch_size'),
                                #gamma=alg_config.getfloat('discount_factor'),
                                #exploration_fraction=alg_config.getfloat('exploration_fraction'),

                                ################################
                                # Add here more hyperparameters if needed, following the above scheme
                                # alg_config['hyperparameter_name']
                                ################################
                                )
    except KeyError or ValueError:
        # Fall back to default policy_kwargs
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
