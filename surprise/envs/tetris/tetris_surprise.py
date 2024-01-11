import numpy as np

from gym import spaces
from surprise.envs.tetris.tetris import TetrisEnv
from surprise.wrappers.base_surprise import BaseSurpriseWrapper
from surprise.buffers.buffers import BernoulliBuffer
import os

import pdb

class EpisodicTetrisSurprise(BaseSurpriseWrapper):
    def __init__(self, episode_length=1000):
        self.episode_length = episode_length
        self.time = 0
        self.done = False
        self.env = TetrisEnv()
        self.buffer = BernoulliBuffer(self.env.observation_space.shape[0]-1)
        super().__init__(self.env, self.buffer, self.episode_length)



        # Grid, Thetas, Time, Next Block
        self.observation_space = spaces.Box(low=0,
                                            high=self.episode_length,
                                            shape=(self.env.height*self.env.width*2+1+1,),
                                            dtype=np.float32)

    def step(self, action):
        assert not self.done, "Can't take actions after done!"

        # Check if done with episode
        if self.time == self.episode_length:
            self.done = True

        self.time += 1
        obs, rew, done, info = super().step(action)

        # Need to see if the underlying env is done, not the wrapped env
        if self.env.done:
            info['num_deaths'] = 1
            self.env.reset()
        else:
            info['num_deaths'] = 0

        return obs, rew, done, info

    def get_obs(self, obs):
        return np.hstack([obs, self.buffer.get_params(), self.time])

    def get_done(self):
        return self.done

    def encode_obs(self, obs):
        # Remove nextBlock before putting on buffer
        obs = obs.copy()
        return obs[:-1]

    def render(self):
        if not os.path.exists("imgs"):
            os.mkdir("imgs", 0o666)
        self.env.render(save='imgs/{:03d}.jpg'.format(env.time))

    def reset(self):
        super().reset()
        self.time = 0
        self.done = False
        return self.get_obs(self.env.get_obs())

env = EpisodicTetrisSurprise()
obs = env.reset()
done = False


while not done:
    obs, rew, done, info = env.step(np.random.randint(12))
    print(info)
    env.render()
    #print(obs)
    #env.env.render(text=True)

