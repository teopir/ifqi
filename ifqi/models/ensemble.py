from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


class Ensemble(object):
    def __init__(self):
        self.models = self.initModel()
        self.modelAdded = True
        self.optimizePrediction=False

    def fit(self, X, y, **kwargs):
        if not hasattr(self, 'sum_'):
            self.sum_ = np.zeros(y.shape)

        sum_ = np.zeros(y.shape)
        for m in self.models[:-1]:
            sum_ += m.predict(X).ravel()
        delta = y - sum_
        ret = self.models[-1].fit(X, delta, **kwargs)
        #self.sum_ += self.models[-1].predict(X).ravel()

        return ret

    def predict(self, x, **kwargs):
        n_samples = x.shape[0]

        if self.optimizePrediction:
            if n_samples > 1:
                if not hasattr(self, 'sumNext_'):
                    self.sumNext_ = np.zeros((n_samples,))
                if self.modelAdded:
                    self.sumNext_ += self.models[-1].predict(x, **kwargs).ravel()
                    self.modelAdded=False

                return self.sumNext_

        output = np.zeros((n_samples,))
        for model in self.models:
            output += model.predict(x).ravel()

        return output

    def adapt(self, iteration):
        self.modelAdded=True
        self.models.append(self.generateModel(iteration))

    def initModel(self):
        model = self.generateModel(0)

        return [model]


class ExtraTreeEnsemble(Ensemble):
    def __init__(self,
                 nEstimators=50,
                 criterion='mse',
                 minSamplesSplit=4,
                 minSamplesLeaf=2):
        self.nEstimators = nEstimators
        self.criterion = criterion
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        super(ExtraTreeEnsemble, self).__init__()

    def generateModel(self, iteration):
        model = ExtraTreesRegressor(n_estimators=self.nEstimators,
                                    criterion=self.criterion,
                                    min_samples_split=self.minSamplesSplit,
                                    min_samples_leaf=self.minSamplesLeaf)

        return model


class MLPEnsemble(Ensemble):
    def __init__(self,
                 nInput=2,
                 nOutput=1,
                 hiddenNeurons=[15],
                 nLayers=1,
                 activation=["relu"],
                 loss='mse',
                 optimizer=None,
                 regularizer=None):
        self.nInput = nInput
        self.nOutput = nOutput
        self.hiddenNeurons = hiddenNeurons
        self.nLayers = nLayers
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        super(MLPEnsemble, self).__init__()

    def generateModel(self, iteration):
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
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


class LinearEnsemble(Ensemble):
    def __init__(self):
        super(LinearEnsemble, self).__init__()

    def generateModel(self, iteration):
        model = LinearRegression()

        return model
