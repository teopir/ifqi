from keras.models import Sequential
from keras.layers import Dense
# Python 2 and 3: forward-compatible
from builtins import range

class IncRegression:
    def __init__(self, n_input=2,
                    n_output=1,
                    hidden_neurons=[15] * 10,
                    n_h_layer_beginning=1,
                    act_function=["sigmoid","sigmoid"] + ["relu"]*8,
                    reLearn=False, optimizer=None):
        self.n_input = n_input
        self.n_output = n_output
        self.optimizer = optimizer
        self.hidden_neurons = hidden_neurons
        self.reLearn = reLearn
        self.n_h_layer_beginning = n_h_layer_beginning
        self.activation = act_function
        
        self.model = self.initModel()
        
    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)
      
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
        
    def adapt(self, iteration=1):
        self.model = self.addLayer()
    
    def addLayer(self):
        new_model = Sequential()
        new_model.add(Dense(self.hidden_neurons[0],
                            input_shape=(self.n_input,),
                            activation=self.activation[0],
                            trainable=self.reLearn,
                            weights=self.model.layers[0].get_weights()))
        i = 1
        for lay in self.model.layers[1:-1]:
            new_model.add(Dense(self.hidden_neurons[i],
                                activation=self.activation[i],
                                trainable=self.reLearn,
                                weights=lay.get_weights()))
            i += 1
        new_model.add(Dense(self.hidden_neurons[i], activation=self.activation[i]))
        new_model.add(Dense(self.n_output, activation='linear'))

        new_model.compile(loss='mse', optimizer=self.optimizer)
        
        return new_model

    def initModel(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        input_shape=(self.n_input,),
                        activation=self.activation[0]))
        for i in range(1, self.n_h_layer_beginning):
            model.add(Dense(self.hidden_neurons[i],
                        activation=self.activation[i]))
        model.add(Dense(self.n_output, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)
        return model