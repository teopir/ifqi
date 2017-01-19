import random, argparse, numpy as np
from Logger import Logger
from joblib import Parallel, delayed
from helpers import flat2list, crop_state, onehot_encode
from tqdm import tqdm
from ifqi import envs


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('--njobs', type=int, default=1,
                    help='number of processes to use, set -1 to use all available cores. Don\'t set this flag if running on GPU.')
parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment to run')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
parser.add_argument('--path', type=str, default='data/model.h5', help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('-e', '--encode', action='store_true', help='save a SARS dataset with the encoded state features')
parser.add_argument('-i', '--images', action='store_true', help='save images of states and a SARS csv with the images\' ids')
args = parser.parse_args()

assert not args.heatmap or (args.heatmap and args.encode), 'Heatmaps can only be generated when the -e flag is set'

logger = Logger(debug=args.debug)
output_csv = 'dataset.csv'

if args.encode:
    from Autoencoder import Autoencoder
    AE = Autoencoder((4, 84, 84), load_path=args.path)
    # Automatically generate headers from the output length of AE.flat_encode
    nb_states = AE.flat_encode(np.expand_dims(np.ones(AE.input_shape), axis=0)).shape[0]
    nb_actions = envs.Atari(args.env).action_space
    output_header = ','.join(
        ['S%s' % i for i in xrange(nb_states)] + ['A%s' % i for i in xrange(nb_actions)] + ['R'] + ['SS%s' % i for i in xrange(nb_states)]
    )
    # Initialize output csv with headers
    logger.to_csv('encoded_' + output_csv, output_header)
if args.images:
    logger.to_csv('images_' + output_csv, 'S,A,R,SS')


def episode(episode_id):
    global args
    env = envs.Atari(args.env)
    action_space = env.action_space.n
    frame_counter = 0

    # Get current state
    state = env.reset()

    # Get encoded features
    if args.encode:
        preprocessed_state = np.expand_dims(np.asarray(crop_state(state)), axis=0)
        encoded_state = AE.flat_encode(preprocessed_state)

    # Save image of state
    if args.images:
        state_id = '%04d_%d' % (episode_id, frame_counter)
        np.save(logger.path + state_id, crop_state(state))

    reward = 0
    done = False

    # Start episode
    while not done:
        frame_counter += 1

        # Select an action
        action = random.randrange(0, action_space)
        # Execute the action, get next state and reward
        next_state, reward, done, info = env.step(action)

        # Get encoded features
        if args.encode:
            preprocessed_next_state = np.expand_dims(np.asarray(crop_state(next_state)), axis=0)
            encoded_next_state = AE.flat_encode(preprocessed_next_state)
            logger.to_csv('encoded_' + output_csv, flat2list([encoded_state, onehot_encode(action, env.action_space), reward, encoded_next_state]))

        # Save image of state
        if args.images:
            next_state_id = '%04d_%d' % (episode_id, frame_counter)
            np.save(logger.path + next_state_id, crop_state(next_state))
            logger.to_csv('images_' + output_csv, [state_id, action, reward, next_state_id])

        # Render environment
        if args.video:
            raw_input()
            env.render()

        # Update state
        state = next_state
        if args.encode:
            encoded_state = encoded_next_state
        if args.images:
            state_id = next_state_id


# Run episodes
print '\nRunning episodes...'
n_jobs = args.njobs
Parallel(n_jobs=n_jobs)(delayed(episode)(eid) for eid in tqdm(xrange(args.episodes)))
