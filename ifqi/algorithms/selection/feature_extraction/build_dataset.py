import gym, random, argparse
from Logger import Logger
from joblib import Parallel, delayed
from Autoencoder import Autoencoder
from helpers import flat2gen
from grid_world.grid_world.envs.gridworld_env import GridWorldEnv

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default='model.h5', help='path to the autoencoder weights file')
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
args = parser.parse_args()

logger = Logger(debug=args.debug)
output_csv = 'dataset.csv'
logger.to_csv(output_csv, 'S0,S1,S2,S3,S4,S5,A0,A1,A2,A3,R,SS0,SS1,SS2,SS3,SS4,SS5')

AE = Autoencoder((90, 160), load_path=args.path)

def episode():
    env = gym.make('GridWorld-v0')
    env.set_grid_size(16, 9) # Optional
    action_space = env.action_space.n
    frame_counter = 0

    state = env.reset()
    encoded_state = AE.encode(state)

    reward = 0
    done = False

    # Start episode
    while not done:
        frame_counter += 1

        # Select an action
        action = random.randrange(0, action_space)
        # Execute the action, get next state and reward
        next_state, reward, done, info = env.step(action)
        encoded_next_state = AE.encode(next_state)

        if args.video:
            raw_input()
            env.render()

        logger.to_csv(output_csv, flat2gen([encoded_state, action, reward, encoded_next_state]))

        # Update state
        state = next_state
        encoded_state = encoded_next_state


    print 'Episode lasted', frame_counter, 'frames'
    # End episode

# Run episodes in parallel
n_jobs = 1 if args.debug else -1
Parallel(n_jobs=n_jobs)(delayed(episode)() for eid in xrange(args.episodes))