from __future__ import print_function
import numpy as np, os, random
from PIL import Image
import pandas as pd


def batch_iterator(dataset_folder, batch_size, nb_epochs, labels=None, shuffle=True):
    """
    Yields batches of length batch_size, randomly created iterating over the dataset nb_epochs times.
    :param labels: use the column with this name as labels and return a x,y dataset
    :param dataset_folder: path to folder containing .png or .jpg images.
    :param batch_size: number of images to yiela at a time. Set to 'all' if you want to use the whole dataset as batch.
    :param nb_epochs: number of times to iterate the dataset.
    :param shuffle: whether to shuffle the data before each epoch.
    :return: an iterator for the batches.
    """

    data = pd.read_csv(dataset_folder + 'images_dataset.csv')
    x = data['S'].as_matrix()
    if labels is not None:
        y = data[labels].as_matrix()
    else:
        y = x

    data_size = len(x)
    batch_size = batch_size if batch_size != 'all' else data_size
    nb_batches_in_epoch = int(data_size / batch_size) + (1 if data_size % batch_size else 0)

    print('Total number of iterations: %d' % (nb_batches_in_epoch * nb_epochs))

    images = []
    labels = []
    for epoch in range(nb_epochs):
        if shuffle:
            # Shuffle data at each epoch
            perm = np.random.permutation(data_size)
            x = x[perm]
            y = y[perm]
        for batch_idx in range(nb_batches_in_epoch):
            images[:] = []  # Empty the list to free up memory
            labels[:] = []
            batch_data = x[batch_idx * batch_size: min((batch_idx + 1) * batch_size, data_size)]
            batch_labels = y[batch_idx * batch_size: min((batch_idx + 1) * batch_size, data_size)]
            for _x, _y in zip(batch_data, batch_labels):
                image = np.load(dataset_folder + _x + '.npy')
                images.append(np.asarray(image))
                labels.append(_y)
            if labels is None:
                yield images
            else:
                yield images, labels


def resize_state(to_resize, new_size=(72,72)):
    """Resizes every image in to_resize to new_size.
    :param to_resize: a numpy array containing a sequence of greyscale images (theano dimension ordering (ch, rows, cols) is assumed)
    :param new_size: the size to which resize the images
    :return: a numpy array with the resized images
    """
    # Iterate over channels (theano dimension ordering (ch, rows, cols) is assumed)
    resized = []
    for image in to_resize:
        data = Image.fromarray(image).resize(new_size)
        resized.append(np.asarray(data))
    return np.asarray(resized).squeeze()


def crop_state(to_crop, keep_top=False):
    """Crops every image in to_crop to a square.
    :param to_crop: a numpy array containing a sequence of greyscale images to crop along axis 1.
    :param keep_top: crop the images keeping the top part.
    :return: the cropped array
    """
    if keep_top:
        return np.split(to_crop, [to_crop.shape[2]], axis=1)[0]
    else:
        return np.split(to_crop, [to_crop.shape[1] - to_crop.shape[2]], axis=1)[1]


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


def flat2list(alist, as_tuple=False, as_set=False):
    """
    :param as_tuple: return a tuple instead of a list
    :param as_tuple: return a set instead of a list
    :param alist: a 2d list
    :return: a flattened version of the list
    """
    output = [i for i in flat2gen(alist)]
    if as_tuple:
        return tuple(output)
    elif as_set:
        return set(output)
    else:
        return output



def onehot_encode(value, nb_categories):
    """
    :param value: discreet value being encoded.
    :param nb_categories: number of possible discreet values being encoded.
    :return: an array of length nb_categories, such that the value-th element equals 1 and all the others 0.
    """
    out = [0] * nb_categories
    out[value] = 1
    return out


def p_load(filename):
    """Loads the numpy object stored as the given filename.

    :param filename: relative path to numpy file.
    :return: the loaded object.
    """
    out = np.load(filename)
    return out


def p_dump(obj, filename):
    """Dumps an object to numpy file.
    :param obj: the object to dump.
    :param filename: the filename to which save the object.
    """
    np.save(filename, obj)
