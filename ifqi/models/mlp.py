from keras.models import Sequential
from keras.layers.core import Dense

class MLP():
    def __init__(self,
                 n_input=2,
                 n_output=1,
                 hidden_neurons=15,
                 n_layers=1,
                 activation="relu",
                 optimizer=None,
                 regularizer=None):
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = n_layers
        self.activation=activation
        self.regularizer = regularizer
        self.model = self.initModel()
        
    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y, **kwargs)
      
    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
        
    def adapt(self, iteration=1):
        pass

    def initModel(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons,
                        input_shape=(self.n_input,),
                        activation=self.activation,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))
        for i in range(1, self.n_layers):
            model.add(Dense(self.hidden_neurons,
                            activation=self.activation,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))
        model.add(Dense(self.n_output,
                        activation='linear',
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))

        model.compile(loss='mse', optimizer=self.optimizer)

        return model