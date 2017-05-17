from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import activity_l1, l2
import numpy as np
import keras.backend.tensorflow_backend as b


class Autoencoder:
    def __init__(self, input_shape, encoding_dim=49, load_path=None, logger=None):
        b.clear_session()  # To avoid memory leaks when instantiating the network in a loop
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.decoding_available = False
        self.dropout_prob = 0.5
        self.logger = logger

        # Build network
        """
        self.inputs = Input(shape=self.input_shape)

        # Encoding layers
        self.encoded = Convolution2D(32, 3, 3, subsample=(3, 3), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.inputs)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)
        self.encoded = Convolution2D(16, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)
        self.encoded = Convolution2D(1, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)

        # self.encoded = Flatten()(self.encoded)
        # self.encoded = Dense(7 * 7, activation='tanh', W_regularizer=l2(), name='encoded')(self.encoded)
        #
        # # Decoding layers
        # self.decoded = Reshape((1, 14, 14))(self.encoded)

        self.decoded = Convolution2D(1, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.encoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(16, 2, 2, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(2, 2), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = UpSampling2D(size=(3, 3), dim_ordering=self.dim_ordering)(self.decoded)
        self.decoded = Convolution2D(self.input_shape[0], 2, 2, border_mode='same', activation='sigmoid', dim_ordering=self.dim_ordering)(self.decoded)
        """

        self.inputs = Input(shape=self.input_shape)
        self.encoded_input = Input(shape=(self.encoding_dim,))

        self.encoded = Dense(self.input_shape[0] / 16)(self.inputs)
        self.encoded = LeakyReLU(alpha=0.01)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)

        self.encoded = Dense(self.input_shape[0] / 32)(self.encoded)
        self.encoded = LeakyReLU(alpha=0.01)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)

        self.encoded = Dense(self.input_shape[0] / 64)(self.encoded)
        self.encoded = LeakyReLU(alpha=0.01)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)

        self.encoded = Dense(self.input_shape[0] / 128)(self.encoded)
        self.encoded = LeakyReLU(alpha=0.01)(self.encoded)
        self.encoded = Dropout(self.dropout_prob)(self.encoded)

        self.encoded = Dense(self.encoding_dim, name='encoded', W_regularizer=l2())(self.encoded)
        self.encoded = LeakyReLU(alpha=0.01)(self.encoded)

        self.decoded = Dense(self.input_shape[0] / 128)(self.encoded)
        self.decoded = LeakyReLU(alpha=0.01)(self.decoded)

        self.decoded = Dense(self.input_shape[0] / 64)(self.decoded)
        self.decoded = LeakyReLU(alpha=0.01)(self.decoded)

        self.decoded = Dense(self.input_shape[0] / 32)(self.decoded)
        self.decoded = LeakyReLU(alpha=0.01)(self.decoded)

        self.decoded = Dense(self.input_shape[0] / 16)(self.decoded)
        self.decoded = LeakyReLU(alpha=0.01)(self.decoded)

        self.decoded = Dense(self.input_shape[0], activation='sigmoid')(self.decoded)

        # Models
        self.autoencoder = Model(input=self.inputs, output=self.decoded)
        self.encoder = Model(input=self.inputs, output=self.encoded)

        # Build decoder model
        if self.decoding_available:
            self.decoding_intermediate = self.autoencoder.layers[-6](self.encoded_input)
            self.decoding_intermediate = self.autoencoder.layers[-5](self.decoding_intermediate)
            self.decoding_intermediate = self.autoencoder.layers[-4](self.decoding_intermediate)
            self.decoding_intermediate = self.autoencoder.layers[-3](self.decoding_intermediate)
            self.decoding_intermediate = self.autoencoder.layers[-2](self.decoding_intermediate)
            self.decoding_output = self.autoencoder.layers[-1](self.decoding_intermediate)
            self.decoder = Model(input=self.encoded_input, output=self.decoding_output)

        # Optimization algorithm
        try:
            self.optimizer = Adam()
        except NameError:
            self.optimizer = RMSprop()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        self.autoencoder.compile(optimizer=self.optimizer, loss=self.contractive_loss, metrics=['accuracy'])

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
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        x = x.reshape(x.shape[0], self.input_shape[0]) # Flatten tensor for dense network
        return self.autoencoder.train_on_batch(x, x)

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
        x = x.reshape(x.shape[0], self.input_shape[0]) # Flatten tensor for dense network
        return self.autoencoder.predict_on_batch(x) * 255  # Restore original scale

    def test(self, x):
        """
        Tests the model on a batch.
        :param x: batch of samples on which to train.
        :return: the metrics of interest as defined in the model (loss, accuracy, etc.)
        """
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        x = x.reshape(x.shape[0], self.input_shape[0]) # Flatten tensor for dense network
        return self.autoencoder.test_on_batch(x, x)

    def encode(self, x):
        """
        Runs the given images through the first half of the autoencoder and returns the encoded features.
        :param x: a batch of samples to encode.
        :return: the encoded batch.
        """
        # Feed input to the model, return encoded images
        x = np.asarray(x).astype('float32') / 255  # Normalize pixels in 0-1 range
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        x = x.reshape(x.shape[0], self.input_shape[0]) # Flatten tensor for dense network
        return self.encoder.predict_on_batch(x)

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
        x = x.reshape(x.shape[0], self.input_shape[0]) # Flatten tensor for dense network
        return np.asarray(self.encoder.predict_on_batch(x)).flatten()

    def decode(self, x):
        """
        Runs the given features through the second half of the autoencoder and returns the reconstructed images.
        :param x: a batch of encoded samples.
        :return: the encoded batch.
        """
        # Feed encoding to the model, return reconstructed images
        if self.decoding_available:
            x = np.asarray(x).astype('float32')
            assert x.shape[1] == self.encoding_dim, \
                'The number of features passed is different from the dimension of the encoding of the network'
            return self.decoder.predict_on_batch(x) * 255
        else:
            print 'Decoding not yet available with this type of network'

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

    def contractive_loss(self, y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        W = K.variable(value=self.autoencoder.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = self.autoencoder.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = 1e-03 * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

        return mse + contractive
