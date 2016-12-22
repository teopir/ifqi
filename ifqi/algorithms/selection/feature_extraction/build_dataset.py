import random, argparse, numpy as np, progressbar
from Logger import Logger
from joblib import Parallel, delayed
from helpers import flat2gen, onehot_encode
from ifqi.envs.gridworld import GridWorldEnv


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/model.h5', help='autoencoder h5 file')
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('-e', '--encode', action='store_true', help='save a SARS dataset with the encoded state features')
parser.add_argument('-i', '--images', action='store_true', help='save images of states with a SARS csv')
parser.add_argument('--heatmap', action='store_true', help='plot correlation matrix of features and coordinates')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
parser.add_argument('--njobs', type=int, default=1, help='number of threads to use (don\'t set this flag if using GPU)')
args = parser.parse_args()

assert args.encode != args.images, 'Set exactly one flag between -i and -e'
assert not args.heatmap or (args.heatmap and args.encode), 'Heatmaps can only be generated when the -e flag is set'

logger = Logger(debug=args.debug)
output_csv = 'dataset.csv'
heatmap_csv = 'heatmap.csv'

if args.encode:
    from Autoencoder import Autoencoder
    logger.to_csv(output_csv, 'S0,S1,S2,S3,S4,S5,A0,A1,A2,A3,R,SS0,SS1,SS2,SS3,SS4,SS5')
    logger.to_csv(heatmap_csv, 'S0,S1,S2,S3,S4,S5,X,Y')
    AE = Autoencoder((1, 64, 96), load_path=args.path)
else:
    logger.to_csv(output_csv, 'S,A,R,S1')


def episode(episode_id):
    global args
    env = GridWorldEnv()
    action_space = env.action_space.n
    frame_counter = 0

    state = env.reset()

    # Get encoded features
    if args.encode:
        preprocessed_state = np.expand_dims(np.expand_dims(np.asarray(state), axis=0), axis=0)
        encoded_state = AE.flat_encode(preprocessed_state)

    # Save image of state
    if args.images:
        state_id = '%04d_%d' % (episode_id, frame_counter)
        state.save(logger.path + state_id + '.png')

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
            preprocessed_next_state = np.expand_dims(np.expand_dims(np.asarray(next_state), axis=0), axis=0)
            encoded_next_state = AE.flat_encode(preprocessed_next_state)
            logger.to_csv(output_csv, [i for i in flat2gen([encoded_state[0], onehot_encode(action, action_space), reward, encoded_next_state[0]])])
            logger.to_csv(heatmap_csv, [i for i in flat2gen([encoded_state[0], env.viewer.char_pos[0], env.viewer.char_pos[1]])])
        # Save image of state
        if args.images:
            next_state_id = '%04d_%d' % (episode_id, frame_counter)
            next_state.save(logger.path + next_state_id + '.png')
            logger.to_csv(output_csv, [state_id, action, reward, next_state_id])

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

    # End episode

# Run episodes
print '\nRunning episodes...'
n_jobs = args.njobs
pb = progressbar.ProgressBar(term_width=50)
Parallel(n_jobs=n_jobs)(delayed(episode)(eid) for eid in pb(xrange(args.episodes)))

# Save heatmap
if args.heatmap:
    # Keep this import sequence as is to correctly create heatmaps on a headless server
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns, pandas as pd

    sns_plot = sns.heatmap(pd.read_csv(logger.path + heatmap_csv).corr())
    sns_plot.get_figure().savefig(logger.path + 'heatmap.png')

