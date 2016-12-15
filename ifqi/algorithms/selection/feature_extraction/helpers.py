import numpy as np, os, random
from PIL import Image


def batch_iterator(dataset_folder, batch_size, nb_epochs, shuffle=True):
    # Generates a batch iterator to train the autoencoder.
    # Batches of length batch_size are randomly created iterating over the dataset nb_epochs times.
    # Set batch size to 'all' if you want to use the whole dataset as batch.

    data = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.png'):
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
	Takes a 2d list and yields its elements in order.
	"""
	for item in alist:
		if isinstance(item, list) or isinstance(item, np.ndarray):
			for subitem in item: yield subitem
		else:
			yield item

def onehot_encode(value, nb_categories):
    out = [0] * nb_categories
    out[value] = 1
    return out
