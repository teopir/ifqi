import gym, random, argparse, time
from Logger import Logger
from joblib import Parallel, delayed
from grid_world.grid_world.envs.gridworld_env import GridWorldEnv

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', action='store_true', help='display video output')
parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
args = parser.parse_args()

logger = Logger(debug=args.debug)
output_csv = 'dataset.csv'
logger.to_csv(output_csv, 'state,action,reward,next_state')

def episode(episode_id):
    env = gym.make('GridWorld-v0')
    env.set_grid_size(16, 9) # Optional
    action_space = env.action_space.n
    frame_counter = 0

    state = env.reset()
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
        next_state_id = '%04d_%d' % (episode_id, frame_counter)
        next_state.save(logger.path + next_state_id + '.png')

        if args.video:
            raw_input()
            env.render()

        logger.to_csv(output_csv, [state_id, action, reward, next_state_id])

        # Update state
        state = next_state
        state_id = next_state_id


    print 'Episode lasted', frame_counter, 'frames'
    # End episode

# Run episodes in parallel
n_jobs = 1 if args.debug else -1
Parallel(n_jobs=n_jobs)(delayed(episode)(eid) for eid in xrange(args.episodes))