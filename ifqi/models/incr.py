from keras.models import Sequential
from keras.layers import Dense

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
    """add a new layer on the stack
       n_neuron: number of neuron of the new layer
        activation_: name of the activation function of the new layer"""
    def addLayer(self, model):
        new_model = Sequential()
        new_model.add(Dense(self.hidden_neurons[0],
                            input_shape=(self.n_input,),
                            activation=self.activation[0],
                            trainable=self.reLearn,
                            weights=model.layers[0].get_weights()))
        i = 1
        for lay in model.layers[1:-1]:
            new_model.add(Dense(self.hidden_neurons[i],
                                activation=self.activation[i],
                                trainable=self.reLearn,
                                weights=lay.get_weights()))
            i += 1
        new_model.add(Dense(self.hidden_neurons[i], self.activation[i]))
        new_model.add(Dense(self.n_output, activation='linear'))

        new_model.compile(loss='mse', optimizer=self.optimizer)
        
        return new_model

    def getModel(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        input_shape=(self.n_input,),
                        activation=self.activation[0]))
        for i in xrange(1, self.n_h_layer_beginning):
            model.add(Dense(self.hidden_neurons[1],
                        activation=self.activation[1]))
        model.add(Dense(self.n_output, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def configureModel(self, model):
        return self.addLayer(model)