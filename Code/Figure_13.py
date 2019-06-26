#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 26, 2019

Figure 13. Evolution of a profile from Atlantic Beach where a sand fence (vertical line) was installed in 2010.
The dashed black line represents MHW (0.34 m, NAVD88).

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


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


def profile_overview(df, color='black'):
    """
    Plot an overlay of all years of data for a single profile. Add
    a line for the sand fence
    """

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    years = ['1997', '1998', '1999', '2000', '2004', '2005', '2010', '2011', '2014', '2016']
    cmap = plt.cm.get_cmap('RdYlBu', len(years))
    xtext, ytext, spacing = 15, 7.5, 0.3

    # Add a grid
    add_grid([ax], xax=True)

    # Add a line for MHW
    ax.axhline(y=0.34, color='black', linestyle='--', linewidth=1, zorder=2)

    # Add a line for the fence
    ax.axvline(x=75, color='black', linewidth=2, zorder=4)

    # Loop through the years an plot
    for ii, year in enumerate(years):

        # Plot the profile
        ax.plot(df['X'], df[year],
                color=cmap(ii / len(years)),
                linewidth=2,
                zorder=2 + ii)

        # Add text labels for the year
        ax.text(x=xtext,
                y=ytext - (ii * spacing),
                s=year,
                color=cmap(ii / len(years)),
                fontname='Arial',
                fontsize=14,
                fontweight='bold',
                zorder=4)

    # Set the x-axis
    ax.set_xlim(left=140, right=0)
    ax.set_xlabel('Cross-Shore Distance (m)',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the y-axis
    ax.set_ylim(bottom=0, top=8)
    ax.set_ylabel('Elevation (m NAVD88)',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the spines
    set_spines([ax], color=color)

    # Make transparent and tight
    tight_and_transparent(fig, [ax], color)

    # Save the figure
    title = 'Figure 13'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


def main():
    """
    Main program function
    """

    # Load the data
    fname = os.path.join('..', 'Data', 'Bogue B Profile 117.csv')
    data = pd.read_csv(fname,
                       delimiter=',',
                       header=None,
                       names=['X', '1997', '1998', '1999', '2000', '2004', '2005', '2010', '2011', '2014', '2016'])

    # Plot the overview
    profile_overview(data, color='black')


if __name__ == '__main__':
    main()
