#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 26, 2019

Figure 14. Evolution of mean natural and fenced dune elevations along Bogue Banks during the post-fencing
period (2010-2016). The shaded area represents the 95% confidence interval.

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

pd.set_option('mode.chained_assignment', None)


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


def fhigh_and_dhigh_v_time(color):
    """
    Make a scatterplot showing the temporal evolution of fenced dune heights and
    natural dune heights. This could be expanded to show all data points but
    given the illustrative nature of the figure, this just plots the mean and
    95% confidence interval for each
    """

    # Set the years
    years = [2010, 2011, 2014, 2016]

    # Set the dhigh (non-fenced areas) values
    nonfenced = [4.2, 4.4, 4.5, 4.7]
    nonfenced_error = [0.08, 0.08, 0.07, 0.06]
    nonfenced_lo = [nonfenced[ii] - nonfenced_error[ii] for ii in range(0, len(nonfenced))]
    nonfenced_hi = [nonfenced[ii] + nonfenced_error[ii] for ii in range(0, len(nonfenced))]

    # Set the dhigh (Fort Macon) values
    fortmacon = [5.2, 5.0, 5.2, 5.8]
    fortmacon_error = [0.06, 0.07, 0.07, 0.06]
    fortmacon_lo = [fortmacon[ii] - fortmacon_error[ii] for ii in range(0, len(fortmacon))]
    fortmacon_hi = [fortmacon[ii] + fortmacon_error[ii] for ii in range(0, len(fortmacon))]

    # Set the dhigh (fenced areas) values
    fenced = [4.1, 4.4, 4.2, 4.3]
    fenced_error = [0.12, 0.12, 0.10, 0.10]
    fenced_lo = [fenced[ii] - fenced_error[ii] for ii in range(0, len(fenced))]
    fenced_hi = [fenced[ii] + fenced_error[ii] for ii in range(0, len(fenced))]

    # Set the fhigh values
    fhigh = [3.4, 3.6, 3.6, 3.9]
    fhigh_error = [0.02, 0.03, 0.02, 0.02]
    fhigh_lo = [fhigh[ii] - fhigh_error[ii] for ii in range(0, len(fhigh))]
    fhigh_hi = [fhigh[ii] + fhigh_error[ii] for ii in range(0, len(fhigh))]

    # Setup the plot
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=300)
    fill_alpha = 0.5
    fenced_line, fenced_face = 'blue', 'cornflowerblue'
    nonfenced_line, nonfenced_face = 'black', 'dimgray'
    fortmacon_line, fortmacon_face = 'darkred', 'red'
    fhigh_line, fhigh_face = 'darkorange', 'peachpuff'

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

    # Plot the Fhigh data
    ax.errorbar(years, fhigh, yerr=fhigh_error, color=fhigh_line, zorder=4)
    ax.fill_between(years, fhigh_hi, fhigh_lo, facecolor=fhigh_face, alpha=fill_alpha,  zorder=2)

    # Add text labels
    xtext, ytext, spacing = 2010, 3.1, 1.5
    labels = ['Fort Macon', 'Non-fenced', 'Behind Fenced', '     Fenced']
    colors = ['darkred', 'dimgray', 'cornflowerblue', 'darkorange']
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
    ax.set_ylim(bottom=3, top=6)
    ax.set_ylabel('D$_{high}$, F$_{high}$ (m, NAVD88)',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the spines
    set_spines([ax], color=color)

    # Make tight and transparent
    tight_and_transparent(fig, [ax], color)

    # Save the figure
    title = 'Figure 14'
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

    # Plot Fhigh and Dhigh versus time
    fhigh_and_dhigh_v_time(color='black')


if __name__ == '__main__':
    main()