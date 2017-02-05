import random
import argparse
import numpy as np
from ifqi import envs
from ifqi.algorithms.selection.feature_extraction.Autoencoder import Autoencoder
from ifqi.algorithms.selection.feature_extraction.helpers import flat2list, crop_state, onehot_encode
from ifqi.algorithms.selection.feature_extraction.Logger import Logger
from joblib import Parallel, delayed
from tqdm import tqdm


def episode_images(episode_id, logger, env_name='BreakoutDeterministic-v3', video=False):
    env = envs.Atari(env_name)
    action_space = env.action_space.n
    frame_counter = 0

    # Get current state
    state = env.reset()

    # Save image of state
    state_id = '%04d_%d' % (episode_id, frame_counter)
    np.save(logger.path + state_id, crop_state(state))

    reward = 0
    done = False

    # Start episode
    ep_output = []
    while not done:
        frame_counter += 1

        # Select an action
        action = random.randrange(0, action_space)
        # Execute the action, get next state and reward
        next_state, reward, done, info = env.step(action)

        # Save image of state
        next_state_id = '%04d_%d' % (episode_id, frame_counter)
        np.save(logger.path + next_state_id, crop_state(next_state))
        ep_output.append([state_id, action, reward, next_state_id])

        # Render environment
        if video:
            env.render()

        # Update state
        state = next_state
        state_id = next_state_id

    return ep_output


def collect_images_dataset(logger, episodes=100, env_name='BreakoutDeterministic-v3', header=None, video=False, n_jobs=-1):
    # Parameters for the episode function
    ep_params = {
        'env_name': env_name,
        'video': video
    }

    # Collect episodes in parallel
    dataset = Parallel(n_jobs=n_jobs)(delayed(episode_images)(eid, logger, **ep_params) for eid in tqdm(xrange(episodes)))
    dataset = np.asarray(flat2list(dataset)) # Each episode is in a list, so the dataset needs to be flattened

    # Return dataset
    if header is not None:
        return np.append([header], dataset, axis=0)
    else:
        return dataset


def episode_encoded(AE, env_name='BreakoutDeterministic-v3', minimum_score=0, onehot=True, video=False):
    env = envs.Atari(env_name)
    action_space = env.action_space.n
    cumulative_reward = 0

    while cumulative_reward <= minimum_score:
        cumulative_reward = 0
        frame_counter = 0

        # Get current state
        state = env.reset()

        # Get encoded features
        preprocessed_state = np.expand_dims(np.asarray(crop_state(state)), axis=0)
        encoded_state = AE.flat_encode(preprocessed_state)

        reward = 0
        done = False

        # Start episode
        ep_output = []
        while not done:
            frame_counter += 1

            # Select an action
            action = random.randrange(0, action_space)
            # Execute the action, get next state and reward
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward

            # Get encoded features
            preprocessed_next_state = np.expand_dims(crop_state(next_state), axis=0)
            encoded_next_state = AE.flat_encode(preprocessed_next_state)

            # Append sars tuple to datset
            actions_to_append = onehot_encode(action, action_space) if onehot else action
            sars_list = [encoded_state, actions_to_append, reward, encoded_next_state, [1 if done else 0] * 2]
            ep_output.append(flat2list(sars_list, as_tuple=True))

            # Render environment
            if video:
                env.render()

            # Update state
            state = next_state
            encoded_state = encoded_next_state

    return ep_output


def collect_encoded_dataset(AE, episodes=100, env_name='BreakoutDeterministic-v3', header=None, onehot=True,
                            minimum_score=0, video=False, n_jobs=-1):
    # Parameters for the episode function
    ep_params = {
        'env_name': env_name,
        'minimum_score': minimum_score,
        'onehot': onehot,
        'video': video
    }

    # Collect episodes in parallel
    dataset = Parallel(n_jobs=n_jobs)(delayed(episode_encoded)(AE, **ep_params) for eid in tqdm(xrange(episodes)))
    dataset = np.asarray(flat2list(dataset)) # Each episode is in a list, so the dataset needs to be flattened

    # Return dataset
    if header is not None:
        return np.append([header], dataset, axis=0)
    else:
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='run in debug mode (no output files)')
    parser.add_argument('-v', '--video', action='store_true', help='display video output')
    parser.add_argument('--njobs', type=int, default=1, help='number of processes to use. Don\'t set this flag w/ GPU.')
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment to run')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes to run')
    parser.add_argument('--path', type=str, default='data/model.h5', help='path to the hdf5 weights file for the AE')
    parser.add_argument('-e', '--encode', action='store_true', help='save a SARS dataset with the encoded features')
    parser.add_argument('-i', '--images', action='store_true', help='save images of states and a SARS csv with ids')
    parser.add_argument('--min-score', type=int, default=0, help='keep episode only if it got more than this score')
    parser.add_argument('--onehot', action='store_true', help='save actions in the dataset with onehot encoding')
    args = parser.parse_args()

    logger = Logger(debug=args.debug)

    if args.encode:
        AE = Autoencoder((4 * 84 * 84,), load_path=args.path)

        # Automatically generate headers from the output length of AE.flat_encode
        nb_states = AE.flat_encode(np.expand_dims(np.ones(AE.input_shape), axis=0)).shape[0]
        nb_actions = envs.Atari(args.env).action_space.n
        actions_header = ['A%s' % i for i in xrange(nb_actions)] if args.onehot else ['A0']
        header = ['S%s' % i for i in xrange(nb_states)] + actions_header + ['R'] + \
                        ['SS%s' % i for i in xrange(nb_states)] + ['Absorbing', 'Finished']

        # Collect episodes
        dataset = collect_encoded_dataset(AE,
                                episodes=args.episodes,
                                env_name=args.env,
                                header=header,
                                onehot=args.onehot,
                                minimum_score=args.min_score,
                                video=args.video,
                                n_jobs=args.njobs)
        output_file = 'encoded_dataset.csv'

    if args.images:
        header = ['S','A','R','SS']
        # Collect episodes
        dataset = collect_images_dataset(logger,
                               episodes=args.episodes,
                               env_name=args.env,
                               header=header,
                               video=args.video,
                               n_jobs=args.njobs)
        output_file = 'images_dataset.csv'
    np.savetxt(logger.path + output_file, dataset, fmt='%s', delimiter=',')