from keras.models import Sequential
from keras.layers.core import Dense

class MLP():
    def __init__(self,
                 n_input=2,
                 n_output=1,
                 hidden_neurons=15,
                 h_layer=1,
                 act_function="relu",
                 optimizer=None,regularizer=None):
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.h_layer = h_layer
        self.act_function=act_function
        self.regularizer = regularizer

    def getModel(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons,
                        input_shape=(self.n_input,),
                        activation=self.act_function,W_regularizer=self.regularizer,b_regularizer=self.regularizer))
        for i in range(1, self.h_layer):
            model.add(Dense(self.hidden_neurons,
                            activation=self.act_function,W_regularizer=self.regularizer,b_regularizer=self.regularizer))
        model.add(Dense(self.n_output,
                        activation='linear',W_regularizer=self.regularizer,b_regularizer=self.regularizer))

        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def configureModel(self, model):
        return model