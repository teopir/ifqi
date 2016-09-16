import gym
import numpy as np

class GymEnv(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.reset()
        self.state_dim = 4
        self.action_dim = 1        
        self.n_actions = 2  
        self.gamma =0.99
        self.absorbing = False
        self.nextState=[0.,0.,0.,0.]
        
    def reset(self):
        self.nextState=[0.,0.,0.,0.]
        self.absorbing = False
        self.env.reset()

    def step(self, action, render=False):
        if render:
            self.env.render()
        nextState, reward, absorbing, info = self.env.step(action) 
        self.nextState = nextState
        self.absorbing = absorbing

        return reward
    
    def isAbsorbing(self):
        return self.absorbing
        
    def isAtGoal(self):
        return False
        
    def getState(self):
        return self.nextState
        
    def runEpisode(self, fqi, render=False):
        """
        This function runs an episode using the regressor in the provided
        object parameter.
        Params:
            fqi (object): an object containing the trained regressor
        Returns:
            - a tuple containing:
                - number of steps
                - J
                - a flag indicating if the goal state has been reached
            - sum of collected reward
        
        """
        J = 0
        t = 0
        test_succesful = 0
        horizon=400
        state_list = []
        action_list = []
        reward_list = []
        while(t < horizon and not self.isAbsorbing()):
            state = self.getState()
            state_list.append(state)
            action, _ = fqi.predict(np.array(state))
            action_list.append(action)
            r = self.step(int(action[0]),render=render)
            reward_list.append(r)
            J += self.gamma ** t * r
            t += 1
            
        if t>=horizon:
            test_succesful = 1
            print("goal reached")
        else:
            print("failure")
            
        state = self.getState()
        state_list.append(state)
        
        s = np.array(state_list)
        a = np.array(action_list)
        s1 = s[1:]
        s = s[:-1]
        t = np.array([[0] * (s.shape[0]-1) + [1]]).T
        #print t
        r = np.array(reward_list)
        #print s.shape
        #print s1.shape
        #print a.shape
        #print t.shape
        sast = np.concatenate((s,a,s1,t),axis=1)
        #print "sast ", sast.shape

        #print "sast ", sast.shape
        #print "r", r.shape
        return J, t, test_succesful, sast, r
        
    def evaluate(self, fqi, render=False):
        """
        This function evaluates the regressor in the provided object parameter.
        This way of evaluation is just one of many possible ones.
        Params:
            fqi (object): an object containing the trained regressor.
        Returns:
            a numpy array containing the average score obtained starting from
            289 different states
        
        """

                
        self.reset()
        J, step, goal, sast, r = self.runEpisode(fqi,render=render)
               
        #(J, step, goal)
        return (J, step, goal, sast, r)
        
                