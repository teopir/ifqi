from environment import Environment

class GymEnv(Environment):
    def __init__(self, env_name):
        self.env = None