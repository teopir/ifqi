import numpy as np, os, random
from PIL import Image


def batch_iterator(dataset_folder, batch_size, nb_epochs, shuffle=True):
    """
    Yields batches of length batch_size, randomly created iterating over the dataset nb_epochs times.
    :param dataset_folder: path to folder containing .png or .jpg images.
    :param batch_size: number of images to yiela at a time. Set to 'all' if you want to use the whole dataset as batch.
    :param nb_epochs: number of times to iterate the dataset.
    :param shuffle: whether to shuffle the data before each epoch.
    :return: an iterator for the batches.
    """

    data = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            data.append(filename)

    data_size = len(data)
    batch_size = batch_size if batch_size != 'all' else data_size
    nb_batches_in_epoch = int(data_size / batch_size) + (1 if data_size % batch_size else 0)

    print 'Total number of iterations: %d' % (nb_batches_in_epoch * nb_epochs)

    images = []
    for epoch in range(nb_epochs):
        if shuffle:
            random.shuffle(data)  # Shuffle data at each epoch
        for batch_idx in range(nb_batches_in_epoch):
            images[:] = []  # Empty the list to free up memory
            batch_data = data[batch_idx * batch_size: min((batch_idx + 1) * batch_size, data_size)]
            for _id in batch_data:
                image = Image.open(dataset_folder + _id).convert('L')
                images.append(np.expand_dims(np.asarray(image), axis=0))
            yield images


def flat2gen(alist):
    """
    :param alist: a 2d list
    :return: a generator for the flattened list
    """
    for item in alist:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            for subitem in item: yield subitem
        else:
            yield item


def flat2list(alist):
    """
    :param alist: a 2d list
    :return: a flattened version of the list
    """
    return [i for i in flat2gen(alist)]


def onehot_encode(value, nb_categories):
    """
    :param value: discreet value being encoded.
    :param nb_categories: number of possible discreet values being encoded.
    :return: an array of length nb_categories, such that the value-th element equals 1 and all the others 0.
    """
    out = [0] * nb_categories
    out[value] = 1
    return out
