from keras.models import Sequential
from keras.layers.core import Dense


class MLP(object):
    def __init__(self,
                 nInput=2,
                 nOutput=1,
                 hiddenNeurons=15,
                 nLayers=1,
                 activation="relu",
                 optimizer=None,
                 regularizer=None):
        self.hiddenNeurons = hiddenNeurons
        self.optimizer = optimizer
        self.nInput = nInput
        self.nOutput = nOutput
        self.nLayers = nLayers
        self.activation = activation
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
        model.add(Dense(self.hiddenNeurons,
                        input_shape=(self.nInput,),
                        activation=self.activation,
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))
        for i in range(1, self.nLayers):
            model.add(Dense(self.hiddenNeurons,
                            activation=self.activation,
                            W_regularizer=self.regularizer,
                            b_regularizer=self.regularizer))
        model.add(Dense(self.nOutput,
                        activation='linear',
                        W_regularizer=self.regularizer,
                        b_regularizer=self.regularizer))

        model.compile(loss='mse', optimizer=self.optimizer)

        return model
