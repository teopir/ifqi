import gym


class Gym(gym.Env):
    """
    The openAI Gym environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, name):
        self.env = gym.make(name)
        self.action_space = self.env.action_space
        self.action_space.values = range(self.action_space.n)
        self.observation_space = self.env.observation_space

        # initialize state
        self.seed()
        self.reset()

    def step(self, action):
        return self.env.step(int(action))

    def get_state(self):
        return self.env.state
