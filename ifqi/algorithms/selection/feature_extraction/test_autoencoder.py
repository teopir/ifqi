from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.envs.gridworld import GridWorldEnv
from helpers import crop_state
from PIL import Image
from ifqi import envs
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/breakout/model.h5', help='path to the hdf5 weights file for the autoencoder')
args = parser.parse_args()

env = envs.Atari('BreakoutDeterministic-v3')
AE = Autoencoder((4, 84, 84), load_path=args.path)

state = env.reset()
preprocessed_state = np.expand_dims(np.asarray(crop_state(state)), axis=0)
encoded_state = AE.encode(preprocessed_state)
flat_encoded_state = AE.flat_encode(preprocessed_state)
predicted_state = AE.predict(preprocessed_state)
predicted_state = predicted_state.reshape(4, 84, 84)
state_img = Image.fromarray(state[0]).convert('L')
pred_img = Image.fromarray(predicted_state[0]).convert('L')

state_img.show()
pred_img.show()
print encoded_state

''' Run this to reconstruct an image from a feature vector
rebuilt = AE.decode(encoded_state)
rebuilt = rebuilt.reshape(1, 48, 48)
reb_img = Image.fromarray(rebuilt[0]).convert('L')
reb_img.show()
'''