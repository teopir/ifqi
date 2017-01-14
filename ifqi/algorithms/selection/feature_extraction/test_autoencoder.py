from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.envs.gridworld import GridWorldEnv
from PIL import Image
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/model/model.h5', help='path to the hdf5 weights file for the autoencoder')
args = parser.parse_args()

env = GridWorldEnv(width=6, height=6, cell_size=8, wall=True, wall_random=True)
AE = Autoencoder((4, 72, 72), load_path=args.path)

state = env.reset()
preprocessed_state = np.expand_dims(np.expand_dims(np.asarray(state), axis=0), axis=0)
encoded_state = AE.encode(preprocessed_state)
flat_encoded_state = AE.flat_encode(preprocessed_state)
predicted_state = AE.predict(preprocessed_state)
predicted_state = predicted_state.reshape(4, 110, 84)
pred_img = Image.fromarray(predicted_state[0]).convert('L')

state.show()
pred_img.show()
print encoded_state

''' Run this to reconstruct an image from a feature vector
rebuilt = AE.decode(encoded_state)
rebuilt = rebuilt.reshape(1, 48, 48)
reb_img = Image.fromarray(rebuilt[0]).convert('L')
reb_img.show()
'''