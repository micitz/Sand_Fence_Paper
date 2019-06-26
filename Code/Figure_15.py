#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 26, 2019

Figure 15. Temporal evolution of the dune system width along Bogue Banks during the post-fencing period (2010-2016).
The dune system for Fort Macon and non-fenced areas consists of a single natural dune, while in fenced areas it
consists of a natural dune fronted by a fenced dune. The shaded areas represent the 95% confidence interval.

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

pd.set_option('mode.chained_assignment', None)


"""
Functions to load and work with the data
"""


def chunks(l, n):
    """
    Split list l into n sized chunks

    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def confidence_intervals(data):
    """
    Calculate the 95% confidence interval
    """

    x_bar = np.nanmean(data)    # Mean value
    s = np.nanstd(data)         # Standard deviation
    n = len(data)               # Sample size

    lo_conf = x_bar - (1.96 * (s / np.sqrt(n)))  # Lower bound of confidence interval
    hi_conf = x_bar + (1.96 * (s / np.sqrt(n)))  # Upper bound of confidence interval

    conf_range = hi_conf - lo_conf  # Size of the 95% confidence interval

    return lo_conf, hi_conf, conf_range


def load_all_data():
    """
    Load all of the data into a dictionary
    """

    data = dict()
    for year in ['2010', '2011', '2014', '2016']:

        data[year] = load_data(int(year))

        # Calculate the dune widths
        data[year]['Dune Width'] = data[year]['x_heel'] - data[year]['x_toe']
        data[year]['Fenced Dune Width'] = data[year]['x_fence_heel'] - data[year]['x_fence_toe']
        data[year]['Fenced Dune System Width'] = data[year]['x_heel'] - data[year]['x_fence_toe']

        # For now, remove all negative widths and volumes, something went wrong with them
        width_condition = data[year]['Fenced Dune Width'] <= 0
        volume_condition = data[year]['Fenced Dune Volume'] <= 0

        data[year]['y_fence_crest'][width_condition] = np.nan
        data[year]['Fenced Dune Width'][width_condition] = np.nan
        data[year]['Fenced Dune Volume'][width_condition] = np.nan

        data[year]['y_fence_crest'][volume_condition] = np.nan
        data[year]['Fenced Dune Width'][volume_condition] = np.nan
        data[year]['Fenced Dune Volume'][volume_condition] = np.nan

        data[year]['Fenced Dune System Width'][data[year]['Fenced Dune System Width'] <= 0] = np.nan

        # Remove instances where the fenced and natural dune crest are not positioned correctly
        crest_condition_1 = data[year]['x_fence_crest'] >= data[year]['x_crest']
        crest_condition_2 = data[year]['y_fence_crest'] >= data[year]['y_crest']

        data[year]['y_fence_crest'][crest_condition_1] = np.nan
        data[year]['Fenced Dune Width'][crest_condition_1] = np.nan
        data[year]['Fenced Dune Volume'][crest_condition_1] = np.nan

        data[year]['y_fence_crest'][crest_condition_2] = np.nan
        data[year]['Fenced Dune Width'][crest_condition_2] = np.nan
        data[year]['Fenced Dune Volume'][crest_condition_2] = np.nan

    data['Fences'] = load_fence_locations(y=0)

    return data


def load_data(year, begin=0, end=None):
    """
    Load the data into Pandas dataframes. Supply the years needed
    as the input argument

    year: Int for the year to look at
    begin: Int for the most westward profile to look at (Default: 0)
    end: Int for the las profile eastward to load
    """

    fname = os.path.join('..', 'Data', 'Morphometrics for Bogue ' + str(year) + '.csv')
    data = pd.read_csv(fname, header=0, delimiter=',')

    data = data.loc[data['Profile No.'] >= begin]

    if end is not None:
        data = data.loc[data['Profile No.'] <= end]

    return data


def load_fence_locations(y, begin=None, end=None):
    """
    Create a dataframe with profile numbers, category,
    and an associated y-value. The y-value is passed to the
    alongshore plotting function and is the y-axis position

    0 - Fort Macon
    1 - Non-fenced
    2 - Fenced

    Called by:
    - Shoreline Change.py
    """

    # Load the data. Can be from any year 2010-2016, results
    # won't change for this application
    data = load_data(2010)

    # Create a fence category column and initialize everything to
    # a value of 0
    data['Category'] = 'darkred'
    data['Number'] = 0

    # Set the rest of the categories
    fort_macon = 2229  # Profiles in from the end where Fort Macon starts
    data['Category'].iloc[:-fort_macon].loc[np.isnan(data['x_fence'])] = 'dimgray'
    data['Category'].iloc[:-fort_macon].loc[~np.isnan(data['x_fence'])] = 'cornflowerblue'

    data['Number'].iloc[:-fort_macon].loc[np.isnan(data['x_fence'])] = 1
    data['Number'].iloc[:-fort_macon].loc[~np.isnan(data['x_fence'])] = 2

    # Add a y-value column
    data['Y'] = y

    # Sort out fences using chunks
    for n in chunks(data[:fort_macon], 5):
        if np.any(data['x_fence'].iloc[n.index] > 0):
            data['Number'].iloc[n.index] = 2
        else:
            data['Number'].iloc[n.index] = 1

    # Just take the needed data
    data = data[['Profile No.', 'Category', 'Number', 'Y']]

    if end is not None:
        data = data.loc[(data['Profile No.'] >= begin) & (data['Profile No.'] <= end)]

    # Return the dataframe
    return data


"""
Functions to make the figure
"""


def tight_and_transparent(fig, axs, color, **kwargs):
    """
    Give the figure a tight layout and a transparent backgorund
    """
    for ax in axs:
        ax.tick_params(colors=color)
        for tick in ax.get_xticklabels():
            tick.set_fontname('Arial')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Arial')

    plt.tight_layout(**kwargs)
    fig.patch.set_color('w')
    fig.patch.set_alpha(0.0)


def set_spines(axs, color):
    """
    Set spines on the lower and left borders. Remove the others
    """

    for ax in axs:

        # Set the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color(color)
        ax.spines['bottom'].set_color(color)

        # Make the tick widths match
        ax.tick_params(width=2)


def add_grid(axs, xax=True):
    """
    Add a grid to the plot. Set xax=False
    if you only want horizontal lines
    """

    for ax in axs:
        ax.grid(color='grey', linewidth=0.25, zorder=0)
        ax.xaxis.grid(xax)


def dune_system_width(data, color='black'):
    """
    Plot the annual dune system width over time
    """

    # Set a list of years
    years = [2010, 2011, 2014, 2016]

    # Set the dhigh (non-fenced areas) values
    nonfenced = [24.0, 25.5, 24.3, 27.6]
    nonfenced_error = [1.45, 1.25, 1.18, 1.32]
    nonfenced_lo = [nonfenced[ii] - nonfenced_error[ii] for ii in range(0, len(nonfenced))]
    nonfenced_hi = [nonfenced[ii] + nonfenced_error[ii] for ii in range(0, len(nonfenced))]

    # Set the dhigh (Fort Macon) values
    fortmacon = [26.1, 24.3, 28.9, 29.5]
    fortmacon_error = [1.26, 1.11, 1.35, 1.11]
    fortmacon_lo = [fortmacon[ii] - fortmacon_error[ii] for ii in range(0, len(fortmacon))]
    fortmacon_hi = [fortmacon[ii] + fortmacon_error[ii] for ii in range(0, len(fortmacon))]

    # Set the values for the fenced areas
    fenced = [np.nanmean(data[str(yr)]['Fenced Dune System Width']) for yr in years]
    fenced_error = [confidence_intervals(data[str(yr)]['Fenced Dune System Width'])[2] for yr in years]
    fenced_lo = [fenced[ii] - fenced_error[ii] for ii in range(0, len(fenced))]
    fenced_hi = [fenced[ii] + fenced_error[ii] for ii in range(0, len(fenced))]

    # Setup the plot
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=300)
    fill_alpha = 0.5
    fenced_line, fenced_face = 'blue', 'cornflowerblue'
    nonfenced_line, nonfenced_face = 'black', 'dimgray'
    fortmacon_line, fortmacon_face = 'darkred', 'red'

    # Add a grid
    add_grid([ax], xax=True)

    # Plot the Dhigh (fenced) data
    ax.errorbar(years, fenced, yerr=fenced_error, color=fenced_line, zorder=4)
    ax.fill_between(years, fenced_hi, fenced_lo, facecolor=fenced_face, alpha=fill_alpha, zorder=2)

    # Plot the Dhigh (non-fenced) data
    ax.errorbar(years, nonfenced, yerr=nonfenced_error, color=nonfenced_line, zorder=4)
    ax.fill_between(years, nonfenced_hi, nonfenced_lo, facecolor=nonfenced_face, alpha=fill_alpha, zorder=2)

    # Plot the Dhigh (Fort Macon) data
    ax.errorbar(years, fortmacon, yerr=fortmacon_error, color=fortmacon_line, zorder=4)
    ax.fill_between(years, fortmacon_hi, fortmacon_lo, facecolor=fortmacon_face, alpha=fill_alpha, zorder=2)

    # Add text labels
    xtext, ytext, spacing = 2010, 20.1, 1.5
    labels = ['Fort Macon', 'Non-fenced', 'Fenced']
    colors = ['darkred', 'dimgray', 'cornflowerblue']
    for ii in range(0, len(labels)):
        ax.text(x=xtext + (ii * spacing),
                y=ytext,
                s=labels[ii],
                color=colors[ii],
                va='bottom',
                fontname='Arial',
                fontsize=12,
                fontweight='bold',
                zorder=2)

    # Set the x-axis
    ax.set_xlabel('Year',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the y-axis
    ax.set_ylim(bottom=20, top=45)
    ax.set_ylabel('Width (m)',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the spines
    set_spines([ax], color=color)

    # Make tight and transparent
    tight_and_transparent(fig, [ax], color)

    # Save the figure
    title = 'Figure 15'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


"""
Run the program
"""


def main():
    """
    Main program function
    """

    # Load all of the data into a dictionary and add sand fence and Fort Macon profiles
    data = load_all_data()

    # Plot the dune system width
    dune_system_width(data, color='black')


if __name__ == '__main__':
    main()