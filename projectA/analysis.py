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



