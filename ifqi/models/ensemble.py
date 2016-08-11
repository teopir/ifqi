from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

from ifqi.models.linear import Linear

class Ensemble(object):
    def __init__(self):
        self.models = self.initModel()
    
    def fit(self, X, y, **kwargs):
        delta = y - self.predictLast(X)
        
        return self.models[-1].fit(X, delta, **kwargs)
      
    def predict(self, x, **kwargs):
        n_samples = x.shape[0]
        output = np.zeros((n_samples,))
        
        for model in self.models:
            output += model.predict(x, **kwargs).ravel()
            
        return output

    def predictLast(self, x, **kwargs):
        n_samples = x.shape[0]
        output = np.zeros((n_samples,))
        
        for model in self.models[:-1]:
            output += model.predict(x, **kwargs).ravel()

        return output
        
    def adapt(self, iteration):
        self.models.append(self.generateModel(iteration))
        
    def initModel(self):
        model = self.generateModel(0)
        
        return [model]
        

class ExtraTreeEnsemble(Ensemble):
    def __init__(self,
                 n_estimators=50,
                 criterion='mse',
                 min_samples_split=4,
                 min_samples_leaf=2):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        super(ExtraTreeEnsemble, self).__init__()
                     
    def generateModel(self, iteration):
        model = ExtraTreesRegressor(n_estimators=self.n_estimators,
                                    criterion=self.criterion,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf)

        return model

        
class MLPEnsemble(Ensemble):
    def __init__(self,
                 n_input=2,
                 n_output=1,
                 hidden_neurons=[15],
                 n_layers=1,
                 activation=["relu"],
                 loss='mse',
                 optimizer=None,
                 regularizer=None):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_neurons = hidden_neurons
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        super(MLPEnsemble, self).__init__()

    def generateModel(self, iteration):
        model = Sequential()
        model.add(Dense(self.hidden_neurons,
                        input_shape=(self.n_input,),
                        activation=self.activation,
                        W_regularizer = self.regularizer,
                        b_regularizer = self.regularizer))
        for i in range(1, self.n_layers):
            model.add(Dense(self.hidden_neurons,
                            activation=self.activation,
                            W_regularizer = self.regularizer,
                            b_regularizer = self.regularizer))
        model.add(Dense(self.n_output,
                        activation='linear',
                        W_regularizer = self.regularizer,
                        b_regularizer = self.regularizer))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model
        
class LinearEnsemble(Ensemble):
    def __init__(self,
                 degree=3):
        self.degree = degree
        super(LinearEnsemble, self).__init__()

    def generateModel(self, iteration):
        model = Linear(self.degree)

        return model