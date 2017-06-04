from keras.models import Sequential, Merge
from keras.layers.core import Dense
from keras.callbacks import  EarlyStopping

"""
Keras MLP wrapper.
With n different models in parallel, in such the way to have the same model complexity of a BFQI model
"""


class FairMLP(object):
    def __init__(self,
                 n_input,
                 n_output,
                 hidden_neurons,
                 activation,
                 optimizer,
                 regularizer=None,
                 early_stopping=False,
                 patience=10,
                 delta_min=0.000001,
                 n_different_models=20):
        assert isinstance(hidden_neurons, list), 'hidden_neurons should be \
            of type list specifying the number of hidden neurons for each \
            hidden layer. The given type is: ' + str(type(hidden_neurons))
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.regularizer = regularizer
        self.early_stopping = early_stopping
        self.delta_min = delta_min
        self.patience = patience
        self.n_different_models=n_different_models
        self.model = self.init_model()

    def fit(self, X, y, **kwargs):
        if self.early_stopping:
            early_s = EarlyStopping(monitor='val_loss', patience=self.patience, min_delta=self.delta_min, verbose=0, mode='auto')
            return self.model.fit(X, y, callbacks=[early_s], validation_split=0.1,**kwargs)
        else:
            return self.model.fit(X, y,**kwargs)

    def reset(self):
        print ("Reset MLP")
        self.model = self.init_model()

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)
        return predictions.ravel()

    def adapt(self, iteration):
        pass

    def init_model(self):

        def tensorAdd(x ):
            ret = x[0]
            for y in x[1:]:
                ret = ret + y
            return ret

        model = [None] * self.n_different_models
        input_layer = Dense(self.hidden_neurons[0],
                            init="glorot_uniform",
                            input_shape=(self.n_input,),
                            activation=self.activation,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer)

        for j in range(self.n_different_models):
            model[j] = Sequential()
            model[j].add(input_layer)
            for i in range(1, len(self.hidden_neurons)):
                model[j].add(Dense(self.hidden_neurons[i],
                                init="glorot_uniform",
                                activation=self.activation,
                                W_regularizer=self.regularizer,
                                b_regularizer=self.regularizer))
            model[j].add(Dense(self.n_output,
                            init="glorot_uniform",
                            activation='linear',
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))

        final_model = Sequential()
        final_model.add(Merge(model))
        final_model.compile(loss='mse', optimizer=self.optimizer)

        return final_model
