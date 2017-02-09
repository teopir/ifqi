from .gymenv import Gym
from gym.envs.registration import register


register(
    id='SwimmerIfqi-v0',
    entry_point='ifqi.envs.swimmer:Swimmer'
)


class Swimmer(Gym):
    """
    The openAI Gym environment.

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self):
        self.horizon = 400
        self.gamma = 0.99

        super(Swimmer, self).__init__('Swimmer-v1')

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            self.env.state = state
            return self.get_state()
