from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
import numpy as np
import keras.backend.tensorflow_backend as b


class Autoencoder:
    def __init__(self, input_shape, load_path=None, logger=None):
        b.clear_session()

        self.dim_ordering = 'th' # (samples, filters, rows, cols)

        self.logger = logger

        # Build network
        # Input layer
        self.inputs = Input(shape=input_shape)  # 64x96

        # Encoding layers
        self.encoded = Convolution2D(128, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.inputs)  # 32x48
        self.encoded = Convolution2D(64, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)  # 16x24
        self.encoded = Convolution2D(32, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)  # 8x12
        self.encoded = Convolution2D(16, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)  # 4x6
        self.encoded = Convolution2D(1, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)  # 2x3

        # Decoding layers
        self.decoded = Convolution2D(1, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)  # 4x6
        self.decoded = Convolution2D(16, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)  # 8x12
        self.decoded = Convolution2D(32, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)  # 16x24
        self.decoded = Convolution2D(64, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)  # 32x48
        self.decoded = Convolution2D(128, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)  # 64x96
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

    def flat_encode(self, x):
        # Feed input to the model, return encoded images flattened
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        return np.asarray(self.encoder.predict_on_batch(x)).flatten()

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
