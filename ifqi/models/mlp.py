from keras.models import Sequential
from keras.layers.core import Dense

"""
Keras MLP wrapper.
"""


class MLP(object):
    def __init__(self,
                 n_input,
                 n_output,
                 hidden_neurons,
                 init,
                 loss,
                 metrics,
                 activation,
                 optimizer,
                 regularizer=None):
        assert isinstance(hidden_neurons, list), 'hidden_neurons should be \
            of type list specifying the number of hidden neurons for each \
            hidden layer.'
        self.hidden_neurons = hidden_neurons
        self.optimizer = optimizer
        self.n_input = n_input
        self.n_output = n_output
        self.init = init
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.regularizer = regularizer
        self.model = self.init_model()

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, x, **kwargs):
        predictions = self.model.predict(x, **kwargs)
        return predictions.ravel()

    def adapt(self, iteration):
        pass

    def init_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_neurons[0],
                        input_shape=(self.n_input,),
                        activation=self.activation,
                        init = self.init,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))
        for i in range(1, len(self.hidden_neurons)):
            model.add(Dense(self.hidden_neurons[i],
                            activation=self.activation,
                            init=self.init,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))
        model.add(Dense(self.n_output,
                        activation='linear',
                        init=self.init,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model

    def count_params(self):
        return self.model.count_params()

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        return self.model.set_weights(w)

    @property
    def layers(self):
        return self.model.layers
