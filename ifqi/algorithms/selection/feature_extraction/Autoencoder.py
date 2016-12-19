from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.regularizers import l2
import numpy as np
import keras.backend.tensorflow_backend as b


class Autoencoder:
    def __init__(self, input_shape, load_path=None, logger=None):
        b.clear_session()

        self.dim_ordering = 'th' # (samples, filters, rows, cols)

        self.logger = logger

        # Input layer
        self.inputs = Input(shape=input_shape)

        # Encoding layers
        self.encoded = Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.inputs)
        self.encoded = AveragePooling2D(pool_size=(2, 2), dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = AveragePooling2D(pool_size=(3, 2), dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = AveragePooling2D(pool_size=(3, 2), dim_ordering=self.dim_ordering)(self.encoded)

        self.encoded = Flatten()(self.encoded)
        self.encoded = Dense(6)(self.encoded)
        self.encoded = LeakyReLU(alpha=0.001)(self.encoded)

        # Decoding layers
        self.decoded = Dense(100, activation='relu')(self.encoded)
        self.decoded = Reshape((1, 5, 20))(self.decoded)
        self.decoded = Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(3, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(3, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(1, 1, 1, border_mode='same', activation='sigmoid', dim_ordering=self.dim_ordering)(self.decoded)

        # Models
        self.autoencoder = Model(input=self.inputs, output=self.decoded)
        self.encoder = Model(input=self.inputs, output=self.encoded)

        # Optimization algorithm
        self.optimizer = Adam()

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
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.train_on_batch(x, x)

    def predict(self, x):
        # Feed input to the model, return encoded and re-decoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.predict_on_batch(x) * 255  # Restore original scale

    def test(self, x):
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.autoencoder.test_on_batch(x, x)

    def encode(self, x):
        # Feed input to the model, return encoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return self.encoder.predict_on_batch(x)

    def save(self, filename=None, append=''):
        # Save the DQN weights to disk
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
            self.autoencoder.save_weights(self.logger.path + f)
        else:
            self.autoencoder.save_weights(f)

    def load(self, path):
        # Load the model and its weights from path
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.autoencoder.load_weights(path)
