from .gymenv import Gym


class CartPole(Gym):
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

        super(CartPole, self).__init__('CartPole-v0')

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            self.env.state = state
            return self.get_state()
