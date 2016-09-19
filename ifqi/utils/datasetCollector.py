import os
import sys
import numpy as np

sys.path.append(os.path.abspath('../'))

from experiment import Experiment


def collect(mdp):
    sast, r = mdp.runEpisode(None, True, False)[3, 4]
    r = np.array(r)

    sarst = np.concatenate((np.concatenate((sast[:, 0:mdp.state_dim], r.T),
                                           axis=1),
                            sast[:, -1 - mdp.state_dim]))

    return sarst

if __name__ == '__main__':
    mdpName = 'LQG1D'
    nEpisodes = 100

    exp = Experiment()
    exp.config['mdp']['mdpName'] = mdpName
    mdp = exp.getMDP(mdpName)
    sarst = collect(mdp)

    sarst.save('../../dataset/' + mdpName)
