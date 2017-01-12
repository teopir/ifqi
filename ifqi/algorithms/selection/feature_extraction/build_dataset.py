import random, argparse, numpy as np, progressbar
from Logger import Logger
from joblib import Parallel, delayed
from helpers import flat2list
from ifqi.envs.gridworld import GridWorldEnv

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('--njobs', type=int, default=1,
                    help='number of processes to use, set -1 to use all available cores. Don\'t set this flag if running on GPU.')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
parser.add_argument('--path', type=str, default='data/model.h5', help='path to the hdf5 weights file for the autoencoder')
parser.add_argument('-e', '--encode', action='store_true', help='save a SARS dataset with the encoded state features')
parser.add_argument('-i', '--images', action='store_true', help='save images of states and a SARS csv with the images\' ids')
parser.add_argument('-c', '--coordinates', action='store_true', help='save a SARS dataset with explicit coordinates')
parser.add_argument('--heatmap', action='store_true', help='save the correlation heatmap of features and coordinates')
args = parser.parse_args()

assert not args.heatmap or (args.heatmap and args.encode), 'Heatmaps can only be generated when the -e flag is set'

logger = Logger(debug=args.debug)
output_csv = 'dataset.csv'
heatmap_csv = 'heatmap.csv'

if args.encode:
    from Autoencoder import Autoencoder

    AE = Autoencoder((1, 48, 48), load_path=args.path)
    # TODO header states must be automatically generated from the output length of AE.flat_encode
    logger.to_csv('encoded' + output_csv, 'S0,S1,S2,S3,S4,S5,S6,S7,S8,X,Y,R,SS0,SS1,SS2,SS3,SS4,SS5,SS6,SS7,SS8')
    logger.to_csv(heatmap_csv, 'S0,S1,S2,S3,S4,S5,S6,S7,S8,X,Y')
if args.images:
    logger.to_csv('images' + output_csv, 'S,A,R,SS')
if args.coordinates:
    logger.to_csv('coordinates' + output_csv, 'pos_X,pos_Y,pos_Wall,act_X,act_Y,R,next_X,next_Y,next_Wall')


def episode(episode_id):
    global args
    env = GridWorldEnv(width=6, height=6, cell_size=8, wall=True, wall_random=True)
    action_space = env.action_space.n
    frame_counter = 0

    # Get current state
    state = env.reset()

    # Get encoded features
    if args.encode:
        preprocessed_state = np.expand_dims(np.expand_dims(np.asarray(state), axis=0), axis=0)
        encoded_state = AE.flat_encode(preprocessed_state)

    # Save image of state
    if args.images:
        state_id = '%04d_%d' % (episode_id, frame_counter)
        state.save(logger.path + state_id + '.png')

    # Save coordinates
    if args.coordinates:
        pos_X = env.viewer.char_pos[0] / env.viewer.cell_size
        pos_Y = env.viewer.char_pos[1] / env.viewer.cell_size
        pos_Wall = list(env.viewer.wall_pos)[0][0] / env.viewer.cell_size

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
            logger.to_csv('encoded' + output_csv, flat2list([encoded_state, env.encode_action(action), reward, encoded_next_state]))
            logger.to_csv(heatmap_csv, flat2list([encoded_state, env.viewer.char_pos[0], env.viewer.char_pos[1]]))

        # Save image of state
        if args.images:
            next_state_id = '%04d_%d' % (episode_id, frame_counter)
            next_state.save(logger.path + next_state_id + '.png')
            logger.to_csv('images' + output_csv, [state_id, action, reward, next_state_id])

        # Save coordinates
        if args.coordinates:
            next_X = env.viewer.char_pos[0] / env.viewer.cell_size
            next_Y = env.viewer.char_pos[1] / env.viewer.cell_size
            next_Wall = list(env.viewer.wall_pos)[0][0] / env.viewer.cell_size
            logger.to_csv('coordinates' + output_csv, flat2list([pos_X, pos_Y, pos_Wall, env.encode_action(action), reward, next_X, next_Y, next_Wall]))

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
        if args.coordinates:
            pos_X = next_X
            pos_Y = next_Y
            pos_Wall = next_Wall


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
