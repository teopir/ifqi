import gym

class GymEnv(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.reset()
        
    def reset(self):
        self.env.reset()

    def step(self, action):
        nextState, reward, absorbing, info = self.env.step(action) 
        self.nextState = nextState
        self.absorbing = absorbing

        return self.nextState, reward
    
    def isAbsorbing(self):
        return self.absorbing
        
    def isAtGoal(self):
        return False
        
    def getState(self):
        return self.nextState
                