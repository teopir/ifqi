import numpy as np

class Actor:
    
    def __init__(self, regressor, epsilon=0.1, n_state=2, n_act=9):
        self.regressor=regressor
        self.epsilon = epsilon
        self.n_act = n_act
        self.n_state = n_state
        
    def exploitAction(self,state):
        q = [0] * self.n_act
        
        if(np.random.rand() < self.epsilon):
            return np.random.randint(0,self.n_act)
            
        actions = np.arange(self.n_act)
        np.random.shuffle(actions)
            
        for a in actions:
            X = np.matrix(state[0:self.n_state] + [a-1])    #-1 just for the inverted pendulum
            q[a] = self.regressor.predict(X)   
            
        return np.argmax(q)
    
    def setEpsilonGreedy(self,epsilon):
        self.epsilon = epsilon
        
    def getEpsilonGreedy(self):
        return self.epsilon  
        