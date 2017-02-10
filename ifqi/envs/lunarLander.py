import gym
import math
import numpy as np
from gym.utils import seeding
from .environment import Environment
from gym import envs
from gym import wrappers


FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class LunarLander(Environment):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self):
        self.gamma = 1.

        self.env = gym.make('LunarLander-v2')
        self.horizon = envs.registry.env_specs["LunarLander-v2"].tags['wrapper_config.TimeLimit.max_episode_steps']

        #self.env = wrappers.Monitor(self.env, './monitor',force=True)
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space
        self.x_random = True

        # initialize state
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            if self.x_random:
                self.env.reset()
                self.env.lander.position.x = np.random.rand()*20.
                self.env.lander.position.y = np.random.rand()*15.
                self.env.lander.linearVelocity.x = np.random.rand()*9.
                self.env.lander.linearVelocity.y = np.random.rand()*13.
                self.env.lander.angle = np.random.rand()*3.
                self.env.lander.angularVelocity = np.random.rand()*20.

                return np.array([
                (self.env.lander.position.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
                (self.env.lander.position.y - (self.env.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
                    self.env.lander.linearVelocity.x*(VIEWPORT_W/SCALE/2)/FPS,
                    self.env.lander.linearVelocity.y*(VIEWPORT_H/SCALE/2)/FPS,
                    self.env.lander.angle,
                20.0*self.env.lander.angularVelocity/FPS,
                1.0 if self.env.legs[0].ground_contact else 0.0,
                1.0 if self.env.legs[1].ground_contact else 0.0
                ])
            else:
                return self.env.reset()
        else:
            self.env.state = state
            return self.get_state()

    def step(self, action):
        return self.env.step(int(action))

    def get_state(self):
        return self.env.state

    def render(self, mode='human', close=False):
        self.env.render()
