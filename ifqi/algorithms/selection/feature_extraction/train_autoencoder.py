import argparse, atexit
from Autoencoder import Autoencoder
from helpers import *
from Logger import Logger


def atexit_handler():
    global AE
    AE.save()


# Register callback to save on exit
atexit.register(atexit_handler)

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='data/', help='')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--batch-size', type=int, default=64, help='')
parser.add_argument('--nb-epochs', type=int, default=3, help='')
args = parser.parse_args()
logger = Logger(debug=args.debug)

AE = Autoencoder((4 * 84 * 84,), logger=logger)

# Create the batch iterator for the images
batches = batch_iterator(args.dataset_dir, batch_size=args.batch_size, nb_epochs=args.nb_epochs)

for idx, batch in enumerate(batches):
    loss, accuracy = AE.train(batch)
    print 'Autoencoder batch %d: loss %f - acc %f' % (idx, loss, accuracy)
