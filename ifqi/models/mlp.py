from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import  EarlyStopping
"""
Keras MLP wrapper.
"""


class MLP(object):
    def __init__(self,
                 n_input,
                 n_output,
                 hidden_neurons,
                 activation,
                 optimizer,
                 regularizer=None,
                 early_stopping=False):
        assert isinstance(hidden_neurons, list), 'hidden_neurons should be \
            of type list specifying the number of hidden neurons for each \
            hidden layer. The given type is: ' + str(type(hidden_neurons))
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.regularizer = regularizer
        self.model = self.init_model()
        self.early_stopping =early_stopping

    def fit(self, X, y, **kwargs):
        if self.early_stopping:
            early_s = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.000001, verbose=0, mode='auto')
            history = self.model.fit(X, y, callbacks=[early_s], validation_split=0.1,**kwargs)
            #print("history_len: ", len(history.history['loss']))
            #print("history_last: ", history.history['loss'][-1])
        else:
            self.model.fit(X, y,**kwargs)

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)
        return predictions.ravel()

    def adapt(self, iteration):
        pass

    def init_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        init="glorot_uniform",
                        input_shape=(self.n_input,),
                        activation=self.activation,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))
        for i in range(1, len(self.hidden_neurons)):
            model.add(Dense(self.hidden_neurons[i],
                            init="glorot_uniform",
                            activation=self.activation,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))
        model.add(Dense(self.n_output,
                        init="glorot_uniform",
                        activation='linear',
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))

        model.compile(loss='mse', optimizer=self.optimizer)

        return model
