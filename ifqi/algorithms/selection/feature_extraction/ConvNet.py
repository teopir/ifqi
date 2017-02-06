from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import activity_l1, l2
import numpy as np
import keras.backend.tensorflow_backend as b


class ConvNet:
    def __init__(self, input_shape, encoding_dim=512, load_path=None, logger=None):
        b.clear_session()  # To avoid memory leaks when instantiating the network in a loop
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.dropout_prob = 0.5
        self.logger = logger

        # Build network
        self.input = Input(shape=self.input_shape)

        self.hidden = Convolution2D(32, 8, 8, border_mode='valid', activation='relu', subsample=(4, 4), dim_ordering='th')(self.input)
        self.hidden = Convolution2D(64, 4, 4, border_mode='valid', activation='relu', subsample=(2, 2), dim_ordering='th')(self.hidden)
        self.hidden = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1), dim_ordering='th')(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(self.encoding_dim, activation='relu')(self.hidden)
        self.output = Dense(1, activation='sigmoid')(self.features)

        # Models
        self.model = Model(input=self.input, output=self.output)
        self.encoder = Model(input=self.input, output=self.features)

        # Optimization algorithm
        try:
            self.optimizer = Adam()
        except NameError:
            self.optimizer = RMSprop()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        # Save the architecture
        if self.logger is not None:
            with open(self.logger.path + 'architecture.json', 'w') as f:
                f.write(self.model.to_json())
                f.close()

    def train(self, x, y):
        """
        Trains the model on a batch.
        :param x: batch of samples on which to train.
        :return: the metrics of interest as defined in the model (loss, accuracy, etc.)
        """
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        y = np.array(y)
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        return self.model.train_on_batch(x, y)

    def predict(self, x):
        """
        Runs the given images through the autoencoder and returns the reconstructed images.
        :param x: a batch of samples on which to predict.
        :return: the encoded and decoded batch.
        """
        # Feed input to the model, return encoded and re-decoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        return self.model.predict_on_batch(x) * 255  # Restore original scale

    def test(self, x, y):
        """
        Tests the model on a batch.
        :param x: batch of samples on which to train.
        :return: the metrics of interest as defined in the model (loss, accuracy, etc.)
        """
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        y = np.array(y)
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        return self.model.test_on_batch(x, y)

    def flat_encode(self, x):
        """
        Runs the given images through the first half of the autoencoder and returns the encoded features in a 1d array.
        :param x: a batch of samples to encode.
        :return: the encoded batch (with flattened features).
        """
        # Feed input to the model, return encoded images flattened
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        return np.asarray(self.encoder.predict_on_batch(x)).flatten()

    def save(self, filename=None, append=''):
        """
        Saves the autoencoder weights to disk (in the run folder if a logger was given, otherwise in the current folder)
        :param filename: custom filename for the hdf5 file.
        :param append: the model will be saved as model_append.h5 if a value is provided.
        """
        # Save the DQN weights to disk
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % self.logger.path + f)
            self.model.save_weights(self.logger.path + f)
        else:
            self.model.save_weights(f)

    def load(self, path):
        """
        Load the model and its weights from path.
        :param path: path to an hdf5 file that stores weights for the model.
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)
