from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
import numpy as np
import keras.backend.tensorflow_backend as b


class Autoencoder:
    def __init__(self, input_shape, load_path=None, logger=None):
        b.clear_session() # To avoid memory leaks when instantiating the network in a loop
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)

        self.logger = logger

        # Build network
        # Input layer
        self.inputs = Input(shape=input_shape)

        # Encoding layers
        self.encoded = Convolution2D(64, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.inputs)
        self.encoded = Convolution2D(32, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Convolution2D(16, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Convolution2D(1, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)

        # Decoding layers
        self.decoded = Convolution2D(1, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(16, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(32, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(64, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid', dim_ordering=self.dim_ordering)(self.decoded)

        # Models
        self.autoencoder = Model(input=self.inputs, output=self.decoded)
        self.encoder = Model(input=self.inputs, output=self.encoded)

        # Optimization algorithm
        self.optimizer = Adadelta()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        self.autoencoder.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Save the architecture
        if self.logger is not None:
            with open(self.logger.path + 'architecture.json', 'w') as f:
                f.write(self.autoencoder.to_json())
                f.close()

    def train(self, x):
        """
        Trains the model on a batch.
        :param x: batch of samples on which to train.
        :return: the metrics of interest as defined in the model (loss, accuracy, etc.)
        """
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.train_on_batch(x, x)

    def predict(self, x):
        """
        Runs the given images through the autoencoder and returns the reconstructed images.
        :param x: a batch of samples on which to predict.
        :return: the encoded and decoded batch.
        """
        # Feed input to the model, return encoded and re-decoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.predict_on_batch(x) * 255  # Restore original scale

    def test(self, x):
        """
        Tests the model on a batch.
        :param x: batch of samples on which to train.
        :return: the metrics of interest as defined in the model (loss, accuracy, etc.)
        """
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.test_on_batch(x, x)

    def encode(self, x):
        """
        Runs the given images through the first half of the autoencoder and returns the encoded features.
        :param x: a batch of samples to encode.
        :return: the encoded batch.
        """
        # Feed input to the model, return encoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.encoder.predict_on_batch(x)

    def flat_encode(self, x):
        """
        Runs the given images through the first half of the autoencoder and returns the encoded features in a 1d array.
        :param x: a batch of samples to encode.
        :return: the encoded batch (with flattened features).
        """
        # Feed input to the model, return encoded images flattened
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return np.asarray(self.encoder.predict_on_batch(x)).flatten()

    def save(self, filename=None, append=''):
        """
        Saves the autoencoder weights to disk (in the run folder if a logger was given, otherwise in the current floder)
        :param filename: custom filename for the hdf5 file.
        :param append: the model will be saved as model_append.h5 if a value is provided.
        """
        # Save the DQN weights to disk
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
            self.autoencoder.save_weights(self.logger.path + f)
        else:
            self.autoencoder.save_weights(f)

    def load(self, path):
        """
        Load the model and its weights from path.
        :param path: path to an hdf5 file that stores weights for the model.
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.autoencoder.load_weights(path)
