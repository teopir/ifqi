from keras.models import Sequential, Model
from keras.layers import Dense, Merge, merge
from keras.utils.layer_utils import layer_from_config
# Python 2 and 3: forward-compatible
#from builtins import range

class IncRegression:
    def __init__(self, n_input=2,
                    n_output=1,
                    hidden_neurons=[15] * 10,
                    n_h_layer_beginning=1,
                    act_function=["sigmoid","sigmoid"] + ["relu"]*8,
                    reLearn=False, optimizer=None, regularizer=None):
        self.n_input = n_input
        self.n_output = n_output
        self.optimizer = optimizer
        self.hidden_neurons = hidden_neurons
        self.reLearn = reLearn
        self.n_h_layer_beginning = n_h_layer_beginning
        self.activation = act_function
        self.regularizer = regularizer
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
                            weights=self.model.layers[0].get_weights(),
                            W_regularizer = self.regularizer,
                            b_regularizer = self.regularizer                        
                            ))
        i = 1
        for lay in self.model.layers[1:-1]:
            new_model.add(Dense(self.hidden_neurons[i],
                                activation=self.activation[i],
                                trainable=self.reLearn,
                                weights=lay.get_weights(),
                                W_regularizer = self.regularizer,
                                b_regularizer = self.regularizer))
            i += 1
        new_model.add(Dense(self.hidden_neurons[i], activation=self.activation[i],
                            W_regularizer = self.regularizer,
                            b_regularizer = self.regularizer))
        new_model.add(Dense(self.n_output, activation='linear'))

        new_model.compile(loss='mse', optimizer=self.optimizer)
        
        return new_model

    def initModel(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        input_shape=(self.n_input,),
                        activation=self.activation[0],
                        W_regularizer = self.regularizer,
                        b_regularizer = self.regularizer))
        for i in range(1, self.n_h_layer_beginning):
            model.add(Dense(self.hidden_neurons[0],
                        activation=self.activation[0],
                        W_regularizer = self.regularizer,
                        b_regularizer = self.regularizer))
        model.add(Dense(self.n_output, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
class MergedRegressor(IncRegression):

    def addLayer(self):
        if not hasattr(self, 'n_steps'):
            self.n_steps = self.n_h_layer_beginning
        nlayers = len(self.model.layers)
        idx = self.n_steps
        for i in range(nlayers):
            self.model.layers[i].trainable = self.reLearn
        
        output = self.model.layers[-2].output
        print(self.n_steps)
        print(self.hidden_neurons[idx])
        new_out = Dense(self.hidden_neurons[idx],
                        activation=self.activation[idx],
                        trainable=True,
                        W_regularizer = self.regularizer,
                        b_regularizer = self.regularizer)(output)
                        
        merged_output = merge([self.model.layers[-1].output, new_out], mode='concat')
        final_loss = Dense(self.n_output, activation='linear')(merged_output)
        
        model = Model(input=[self.model.layers[0].input], output=final_loss)
        model.compile(loss='mse', optimizer=self.optimizer)
        self.n_steps += 1
        return model

