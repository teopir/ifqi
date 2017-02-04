from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.envs.gridworld import GridWorldEnv
from helpers import crop_state
from PIL import Image
from ifqi import envs
import argparse, random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/breakout/model.h5', help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('--original', action='store_true')
parser.add_argument('--rebuilt', action='store_true')
parser.add_argument('--convmap', action='store_true')
args = parser.parse_args()

env = envs.Atari('BreakoutDeterministic-v3')
AE = Autoencoder((4 * 84 * 84,), load_path=args.path)

state0 = env.reset()

# Move base to the left
state0, reward, done, info = env.step(3)
state00, reward, done, info = env.step(3)  # Base moves one step wrt state 0

# Start the game
state1, reward, done, info = env.step(1)

# Keep base to the left and move only the ball
for i in range(10):
    state1, reward, done, info = env.step(3)
state2, reward, done, info = env.step(3)  # Ball moves one step wrt state 1

# Keep base to the left and move only the ball
for i in range(19):  #
    state3, reward, done, info = env.step(3)
state4, reward, done, info = env.step(3)  # Ball moves one step wrt state 3 and breaks wall

preprocessed_state0 = np.expand_dims(np.asarray(crop_state(state1)), axis=0)
encoded_state0 = AE.encode(preprocessed_state0)
flat_encoded_state0 = AE.flat_encode(preprocessed_state0)
predicted_state0 = AE.predict(preprocessed_state0)
predicted_state0 = predicted_state0.reshape(4,84, 84)
conv_map0 = flat_encoded_state0.reshape(7,7)

preprocessed_state00 = np.expand_dims(np.asarray(crop_state(state00)), axis=0)
encoded_state00 = AE.encode(preprocessed_state00)
flat_encoded_state00 = AE.flat_encode(preprocessed_state00)
predicted_state00 = AE.predict(preprocessed_state00)
predicted_state00 = predicted_state00.reshape(4,84, 84)
conv_map00 = flat_encoded_state00.reshape(7,7)

preprocessed_state1 = np.expand_dims(np.asarray(crop_state(state1)), axis=0)
encoded_state1 = AE.encode(preprocessed_state1)
flat_encoded_state1 = AE.flat_encode(preprocessed_state1)
predicted_state1 = AE.predict(preprocessed_state1)
predicted_state1 = predicted_state1.reshape(4,84, 84)
conv_map1 = flat_encoded_state1.reshape(7,7)

preprocessed_state2 = np.expand_dims(np.asarray(crop_state(state2)), axis=0)
encoded_state2 = AE.encode(preprocessed_state2)
flat_encoded_state2 = AE.flat_encode(preprocessed_state2)
predicted_state2 = AE.predict(preprocessed_state2)
predicted_state2 = predicted_state2.reshape(4,84, 84)
conv_map2 = flat_encoded_state2.reshape(7,7)

preprocessed_state3 = np.expand_dims(np.asarray(crop_state(state3)), axis=0)
encoded_state3 = AE.encode(preprocessed_state3)
flat_encoded_state3 = AE.flat_encode(preprocessed_state3)
predicted_state3 = AE.predict(preprocessed_state3)
predicted_state3 = predicted_state3.reshape(4,84, 84)
conv_map3 = flat_encoded_state3.reshape(7,7)

preprocessed_state4 = np.expand_dims(np.asarray(crop_state(state4)), axis=0)
encoded_state4 = AE.encode(preprocessed_state4)
flat_encoded_state4 = AE.flat_encode(preprocessed_state4)
predicted_state4 = AE.predict(preprocessed_state4)
predicted_state4 = predicted_state4.reshape(4,84, 84)
conv_map4 = flat_encoded_state4.reshape(7,7)

print '\nOnly base moves (% change between states)'
print(np.array_str(conv_map0 / conv_map00, precision=2))
print '\nOnly ball moves (% change between states)'
print(np.array_str(conv_map1 / conv_map2, precision=2))
print '\nOnly ball moves and breaks wall (% change between states)'
print(np.array_str(conv_map3 / conv_map4, precision=2))
print ''

if args.convmap:
    Image.fromarray(conv_map0 * 255).convert('L').show()
    Image.fromarray(conv_map00 * 255).convert('L').show()
    Image.fromarray(conv_map1 * 255).convert('L').show()
    Image.fromarray(conv_map2 * 255).convert('L').show()
    Image.fromarray(conv_map3 * 255).convert('L').show()
    Image.fromarray(conv_map4 * 255).convert('L').show()

if args.original:
    Image.fromarray(state0[3]).convert('L').show()
    Image.fromarray(state00[3]).convert('L').show()
    Image.fromarray(state1[3]).convert('L').show()
    Image.fromarray(state2[3]).convert('L').show()
    Image.fromarray(state3[3]).convert('L').show()
    Image.fromarray(state4[3]).convert('L').show()

if args.rebuilt:
    Image.fromarray(predicted_state0[3]).convert('L').show()
    Image.fromarray(predicted_state00[3]).convert('L').show()
    Image.fromarray(predicted_state1[3]).convert('L').show()
    Image.fromarray(predicted_state2[3]).convert('L').show()
    Image.fromarray(predicted_state4[3]).convert('L').show()
    Image.fromarray(predicted_state4[3]).convert('L').show()

''' Run this to reconstruct an image from a feature vector
rebuilt = AE.decode(encoded_state)
rebuilt = rebuilt.reshape(1, 48, 48)
reb_img = Image.fromarray(rebuilt[0]).convert('L')
reb_img.show()
'''