import numpy as np
import pandas as pd
import simpy
import math


from numpy.random import default_rng
from collections import defaultdict


rng = np.random.default_rng(1361)
STONES = 10


def get_dfs_counts(prediction):
    """
    Get the count of Drought, Flood and Safe from an array of events

    Input must be something like this ['Drought', ' Flood', 'Safe' ...]
    """
    choice, counts = np.unique(prediction, return_counts=True)
    temp = dict(zip(choice, counts))

    d = temp['Drought'] if 'Drought' in temp else 0
    f = temp['Flood'] if 'Flood' in temp else 0
    s = temp['Safe'] if 'Safe' in temp else 0

    return d, f, s


def get_rand_pmf():
    """
    Get a random PMF
    Output will be an array with shape (3,) with sum equal 1
    """
    pmf = rng.dirichlet(np.ones(3), size=1).squeeze()

    return pmf


def check_pmf(pmf):
    """
    Ensure input pmf has sum equal to 1
    """
    pmf_sum = round(pmf.sum(), 3)
    if (pmf < 0).any():
        raise ValueError('Some values of pmf is negative')
    elif (pmf > 1).any():
        raise ValueError('Some values of pmf is more than 1')
    elif (pmf_sum < 0.999) or (pmf_sum > 1.001):
        raise ValueError('pmf sum not 1')


def get_all_strategies(num_stones):
    """
    Get all possible ways to split stones.
    Output will be an array of shape (3, -1) and it is
    arranged as such (drought insurance, flood insurance, safe investment)
    """
    di, fi, si = [], [], []

    for d in range(num_stones + 1):
        for f in range(num_stones - d + 1):
            s = num_stones - d - f
            di.append(d)
            fi.append(f)
            si.append(s)

    return np.array([di, fi, si])


def default_scoring(d, f, s, all_strats):
    """
    Scoring system is a very basic one.
    If drought insurance >= drought and flood insurance >= flood.
    player will get a score equal to their safe investment or else they
    will get 0
    """
    is_correct = (all_strats[0] >= d) & (all_strats[1] >= f)
    return list(all_strats[2] * is_correct)


# SIMULATION
def one_decade(env, env_pmf=None, all_strats=get_all_strategies(STONES),
               num_sim=10000, log={}, verbose=False):
    """
    Each simulation will get the occurrences of drought, flood and safe over a period of one decade

    `log` keeps track of what each 'player' predicted and what the actual events were.
    This is for manipulation of data.
    Each row of `log` -> [Player, Iteration, Drought Insurance,
                          Flood Insurance, Investment, Drought,
                          Flood, Safe, Prosperity]

    all_strats is a 2d numpy array with shape(-1, 3)
    all_strats[:, 0] is Drought Insurance
    all_strats[:, 1] is Flood Insurance
    all_strats[:, 2] is Investment
    """
    log.clear()

    num_strats = len(all_strats[0])

    #initialise records:
    scores = np.zeros(num_strats, dtype=int)
    dt, ft, st = 0, 0 ,0

    #initialise log:
    keys = ('Player', 'Iteration', 'Drought Insurance', 'Flood Insurance', 'Investment', 'Drought', 'Flood', 'Safe', 'Prosperity')
    log.update({k: [] for k in keys})

    # Get a random pmf if no pmf is provided
    env_pmf = get_rand_pmf() if env_pmf is None else env_pmf
    check_pmf(env_pmf)

    # number of simulation
    while env.now < num_sim:

        # Add to log
        log['Iteration'].extend([env.now] * num_strats)

        # Get the occurrences of D, F, S over one decade
        actual = rng.choice(['Drought', 'Flood', 'Safe'], 10, p=env_pmf)
        d, f, s = get_dfs_counts(actual)

        # add to actual log
        log['Drought'].extend([d] * num_strats)
        log['Flood'].extend([f] * num_strats)
        log['Safe'].extend([s] * num_strats)

        # add to player log
        log['Player'].extend([*range(0, num_strats)])
        log['Drought Insurance'].extend(all_strats[0])
        log['Flood Insurance'].extend(all_strats[1])
        log['Investment'].extend(all_strats[2])

        # add to event counts
        dt += d; ft += f; st += s

        # tabulate the score
        log['Prosperity'].extend(default_scoring(d, f, s, all_strats))

        # wait
        yield env.timeout(1)

    if verbose: print(f'Total: {dt+ft+st:,}\nDroughts: {dt:,}, Floods: {ft:,}, Safe: {st:,}')
