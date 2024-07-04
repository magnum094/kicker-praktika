from src.sb3.stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecEnvWrapper, VecEnv
from src.sb3.stable_baselines3.common.vec_env.vec_pbrs import VecPBRSWrapper
from src.sb3.stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from src.sb3.stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from gymnasium import Wrapper


class VecLingeringRewardDistance(VecEnvWrapper):
    total_timesteps = 1000
    goal_post = [1.296, 0, 0.2]
    def __init__(self, venv: VecEnv, goal_post, total_timesteps):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.total_timesteps = total_timesteps
        self.goal_post = goal_post        
        self.venv = venv

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

            

        for i in range(len(rewards)):
                distance_ball_goal = 0
                for j in range(len(infos)):
                    distance_ball_goal = sum([(infos[j]["ball_position"][i] - self.goal_post[i])**2 for i in range(3)])**0.5    
                lingering_reward = 1*distance_ball_goal/self.total_timesteps
                rewards[i] = rewards[i] - lingering_reward/self.total_timesteps

        return observations, rewards, dones, infos 


class VecLingeringRewardTime(VecEnvWrapper):
    total_timesteps = 1000
    def __init__(self, venv: VecEnv, total_timesteps):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.total_timesteps = total_timesteps
        self.venv = venv


    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

        for i in range(len(rewards)):
                rewards[i] = rewards[i] - 1/self.total_timesteps

        return observations, rewards, dones, infos 

class VecScaleNegReward(VecEnvWrapper):
    scale_factor = 1

    def __init__(self, venv: VecEnv, scale_factor):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.scale_factor = scale_factor
        self.venv = venv

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        for i in range(len(rewards)):
            if rewards[i] < 0:
                rewards[i] = rewards[i] * self.scale_factor
        return observations, rewards, dones, infos

class VecScalePositivReward(VecEnvWrapper):
    scale_factor = 1

    def __init__(self, venv: VecEnv, scale_factor):
        VecEnvWrapper.__init__(self, venv, venv.observation_space, venv.action_space)
        self.scale_factor = scale_factor
        self.venv = venv

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        for i in range(len(rewards)):
            if rewards[i] > 0:
                rewards[i] = rewards[i] * self.scale_factor
        return observations, rewards, dones, infos

class LingeringReward(Wrapper):
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        goal_post = [1.296, 0, 0.2]
        distance_ball_goal = sum([(info["ball_position"][i] - goal_post[i])**2 for i in range(3)])**0.5
        lingering_reward = 0.0001*distance_ball_goal
        return next_state, reward-lingering_reward, terminated, truncated, info