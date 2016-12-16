import argparse, progressbar
from Autoencoder import Autoencoder
from helpers import *
from Logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='data/', help='')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--batch-size', type=int, default=64, help='')
parser.add_argument('--nb-epochs', type=int, default=5, help='')
args = parser.parse_args()
logger = Logger(args.debug)

AE = Autoencoder((1, 90, 160), logger=logger)
batches = batch_iterator(args.dataset_path, batch_size=args.batch_size, nb_epochs=args.nb_epochs)

for idx, batch in enumerate(batches):
    loss, accuracy = AE.train(batch)
    print 'Autoencoder batch %d: loss %f - acc %f' % (idx, loss, accuracy)


AE.save()

