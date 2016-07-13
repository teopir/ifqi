import gym

class GymEnv(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.reset()
        
    def reset(self):
        self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action) 
        self.nextState = observation
        self.abs = done
        return reward
    
    def isAbsorbing(self):
        return self.abs
        
    def isAtGoal(self):
        return False
        
    def getState(self):
        return self.nextState
                