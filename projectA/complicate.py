import numpy as np
import pandas as pd


from numpy.random import default_rng


rng = np.random.default_rng(1361)


def get_dmg(num_iter, exp):
    """
    Damage cause is taken to be follow a exponential distribution

    Create a lookup table so that damages can be mapped easily
    df must contain column `Iteration`

    Lookup table will be of shape (-1, 10)
    Each row denote a single iteration
    For a single row, each value denote the damage of a single drought
    or flood

    For example, in iteration 100, there are 3 floods.
    We will index the 100th row from the lookup table.
    Since each value in that row denote the damage of a single drought,
    we will sum up the first 3 elements of that row to obtain the total
    flood damage
    """
    lookup = rng.exponential(exp, (num_iter, 10))

    def get_total_dmg(iteration, event_count):
        # Using lambda will change them to float
        iteration = int(iteration)
        event_count = int(event_count)
        return round(np.sum(lookup[iteration, :event_count]), 2)

    return get_total_dmg


def get_investment(num_iter, mean):
    """
    Investment return per year is taken follow a normal distribution
    Investment is compounded annually

    Create a lookup table so that investment can be mapped easily
    df must contain column `Iteration`

    Lookup table will be of shape (-1, 10)
    Each row denote a single iteration
    For a single row, each value denote the return of a single year

    For example, in iteration 100, a player invested 6 stones
    We will index the 100th row from the lookup table.
    Since each value in that row denote the return of a single year,
    we will multiply the all the elements in the row then multiply it by
    6. We will get the total investment returns for that player
    """
    lookup = rng.normal(mean, 0.1, (num_iter, 10))

    def get_returns(iteration, investment):
        # Using lambda will change them to float
        returns = lookup[int(iteration)]

        return round(np.prod(returns) * investment, 2)

    return get_returns


def account_dmg_investment(actual_df, drought_exp, flood_exp, returns_mean):
    """
    Account for drought and flood damages and investment returns
    A new prospoerity score will be calculated based on that

    To calculate the prosperity score,
    if drought insurance >= drought damage
    and flood insurance >= flood damage, the prosperity score will be
    equal to the investment returns else it will be 0

    Drought and flood damages is taken to follow an exp distribution
    Investment return per year is taken to follow a normal distribution
    """
    # make a copy
    df = actual_df.copy()

    num_iter = len(df.Iteration.unique())

    # Get functions to calculate damages and returns
    get_drought_damage = get_dmg(num_iter, drought_exp)
    get_flood_damage = get_dmg(num_iter, flood_exp)
    get_returns = get_investment(num_iter, returns_mean)

    # Calculate damages and returns
    df['Drought Damage'] = df.apply(lambda x: get_drought_damage(x.Iteration, x.Drought), axis=1)
    df['Flood Damage'] = df.apply(lambda x: get_flood_damage(x.Iteration, x.Flood), axis=1)
    df['Investment Gains'] = df.apply(lambda x: get_returns(x.Iteration, x.Investment), axis=1)

    # Calculate new Prosperity scores
    is_correct = ((df['Drought Insurance'] >= df['Drought Damage']) &
                  (df['Flood Insurance'] >= df['Flood Damage']))
    df['Prosperity'] = df['Investment Gains'] * is_correct

    # Reorder df
    df = df[['Player', 'Iteration', 'Drought Insurance', 'Flood Insurance',
             'Investment', 'Investment Gains', 'Drought', 'Drought Damage',
             'Flood', 'Flood Damage', 'Safe', 'Prosperity']]

    return df
