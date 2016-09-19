import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))


def loadIfqiDataset(path, nEpisodes=None):
    """
    This function loads the dataset with the given number of episodes
    from a .npy file saved by the main function of this module.

    Args:
        - path (string): the path of the file containing the dataset
        - nEpisodes (int): the number of episodes to consider inside the
                           dataset

    Returns:
        - the loaded dataset

    """
    assert nEpisodes > 0, "Number of episodes to compute must be greater than \
                           zero."
    data = np.load(path)
    episodeStarts = np.argwhere(data[:, 0] == 1).ravel()

    if nEpisodes is not None:
        assert np.size(episodeStarts) >= nEpisodes

        return data[:episodeStarts[nEpisodes - 1] + 1, 1:]
    return data[:, 1:]


from experiment import Experiment


def collect(mdp, nEpisodes):
    """
    Collection of the dataset using the given environment and number
    of episodes.

    Args:
        - mdp (object): the environment to consider
        - nEpisodes (int): the number of episodes

    Returns:
        - the dataset

    """
    data = mdp.collect()

    for i in xrange(nEpisodes - 1):
        data = np.append(data, mdp.collect(), axis=0)

    return data

if __name__ == '__main__':
    mdpName = 'CarOnHill'
    fileName = 'coh'
    nEpisodes = 1000

    exp = Experiment()
    exp.config['mdp'] = dict()
    exp.config['mdp']['mdpName'] = mdpName
    mdp = exp.getMDP()
    data = collect(mdp, nEpisodes)

    np.save('../../dataset/' + mdpName + '/' + fileName, data)
