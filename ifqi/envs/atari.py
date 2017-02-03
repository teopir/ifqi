import gym
import numpy as np
from gym.utils import seeding
from PIL import Image


class Atari(gym.Env):
    """
    The Atari environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, name='PongDeterministic-v3'):
        self.IMG_SIZE = (84, 110)
        self.gamma = 0.99

        self.env = gym.make(name)
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        # initialize state
        self.seed()
        self.reset()

    def reset(self, state=None):
        state = self._preprocess_observation(self.env.reset())
        self.env.state = np.array([state, state, state, state])
        return self.get_state()

    def step(self, action):
        current_state = self.get_state()
        obs, reward, done, info = self.env.step(int(action))
        obs = self._preprocess_observation(obs)
        self.env.state = self._get_next_state(current_state, obs)
        return self.get_state(), reward, done, info

    def get_state(self):
        return self.env.state

    def _preprocess_observation(self, obs):
        image = Image.fromarray(obs, 'RGB').convert('L').resize(self.IMG_SIZE)
        return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                                   image.size[0])  # Convert to array and return

    def _get_next_state(self, current, obs):
        # Next state is composed by the last 3 images of the previous state and the new observation
        return np.append(current[1:], [obs], axis=0)

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)
