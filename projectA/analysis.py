import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def print_top_players(df, n=5):
    """
    Print the top n player for that round.

    Note that scores printed will be based on column name `Prosperity`.
    `Prosperity` must be overwritted if there is a new way to calculate
    scores.
    In addition, columns that are required are `'Drought Insurance`,
    `Flood Insurance` and `Investment`.
    Without the 4 columns, function will not work
    """
    sum_prosperity = df.groupby('Player').Prosperity.sum()
    avg_prosperity = df.groupby('Player').Prosperity.mean()
    top_index = avg_prosperity.sort_values(ascending=False).index[:n]

    di = df['Drought Insurance']
    fi = df['Flood Insurance']
    inv = df['Investment']
    rank = 1
    for i in top_index:
        print(f'Rank {rank:2.0f}: ({di[i]:2.0f}, {fi[i]:2.0f}, {inv[i]:2.0f}) | Total Prosperity: {sum_prosperity[i]:.2f}, Average Prosperity: {avg_prosperity[i]:.2f}')
        rank += 1

    print(f'\n(i, j , k) -> (Drought insurance, Flood Insurance, Investment)')


def quick_plot(dfs, titles=None, max_row=2, suptitle=None):
    """
    Plot one heatmap per df.

    max_row is max per row.

    Heatmap will have Drought Insurance as x axis and Flood Insurance as
    y axis. Intensity of each square correspond to the average scores
    gotten from the simulation.
    """

    # deal with single df input
    if type(dfs) == pd.core.frame.DataFrame:
        dfs = [dfs]

    # deal with single title input
    if type(titles) == str:
        titles = [titles]

    num_df = len(dfs)

    # Get rows and cols
    if num_df > max_row:
        cols = max_row
        rows = math.ceil(num_df/max_row)
    else:
        cols = num_df
        rows = 1

    # set figure size properly
    plt.figure(figsize=(21, 6*rows))

    # set super title
    if suptitle:
        plt.suptitle(suptitle, fontsize=21, fontweight='bold')

    # plot all dfs
    for idx in range(num_df):
        pv = dfs[idx].pivot_table(values='Prosperity', index='Flood Insurance', columns='Drought Insurance')

        # Plot subplot, heatmap and colorbar
        plt.subplot(rows, cols, idx+1)
        plt.imshow(pv, cmap="Oranges")
        plt.colorbar().set_label('Expected Prosperity')

        # Set subplot details
        plt.xticks(range(len(pv)), pv.columns)
        plt.yticks(range(len(pv)), pv.index)
        plt.xlabel('Drought Insurance')
        plt.ylabel('Flood Insurance')

        if titles:
            plt.title(titles[idx],  fontsize=16, fontweight='roman')

    plt.show()


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

