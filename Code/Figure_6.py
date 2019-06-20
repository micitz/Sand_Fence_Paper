#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 20, 2019

Figure 6: Examples of Automorph output. A) Profile from Fort Macon
where sand fences are present. B) Profile that includes a sand fence
(red marker) and a fenced dune located seaward of the natural foredune.

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


"""
Function(s) to load and format the data
"""


def load_data(p_type, index):
    """
    Load the profile and morphometrics

    - p_type: String denoting "Fenced" or "Natural"
    - index: Profile number to use
    """

    # Load the profile
    profile_fname = os.path.join('..', 'Data', f'{p_type} Profile.csv')
    profile = pd.read_csv(profile_fname,
                          delimiter=',',
                          header=None,
                          names=['X', 'Y'])

    # Load the morphometrics
    morpho_fname = os.path.join('..', 'Data', 'Morphometrics for Bogue 2014.csv')
    morpho = pd.read_csv(morpho_fname, delimiter=',', header=0)
    morpho = morpho.loc[[index - 1]]

    return profile, morpho


"""
Functions to make and style
the figure
"""


def add_labels(ax, features, colors):
    """
    Add labels to the figure pointing out
    what the morphometrics are
    """

    # Set the capitalization and remove "_" from
    # the feature names
    for ii, feature in enumerate(features):
        if '_' in feature:
            feature = feature.replace('_', ' ')
        if ii == 0:
            features[ii] = feature.upper()
        else:
            features[ii] = feature.title()

    # Add the labels
    x_text, y_text, spacing, counter = -5.00, 7.00, 0.75, 0
    for ff, cc in zip(features, colors):
        ax.text(x=x_text,
                y=y_text - (counter * spacing),
                s=ff,
                color=cc,
                fontname='Arial',
                fontsize=14,
                fontweight='bold')
        counter += 1


def add_profile(ax, x, y):
    """
    Plot the profile and color it in
    with a sandy color
    """

    ax.plot(x, y, color='black', linewidth=1.5, zorder=3)
    ax.fill_between(x=x,
                    y1=y,
                    y2=0,
                    color=(0.93, 0.79, 0.69),
                    edgecolor='black',
                    linewidth=2,
                    zorder=3)


def add_water(ax, x):
    """
    Add a water colored patch to
    the figure between 0m and 0.34m
    """

    ax.axhline(y=0.34, color='black', linewidth=1.5, zorder=2)
    ax.fill_between(x=x,
                    y1=0.34,
                    y2=0,
                    color=(0.51, 0.90, 0.85),
                    edgecolor='black',
                    linewidth=2,
                    zorder=2)


def plot_features(ax, morpho, features, colors):
    """
    Plot the dune morphometrics
    on the profile
    """

    for feature, color in zip(features, colors):
        ax.scatter(x=morpho[f'local_x_{feature}'],
                   y=morpho[f'y_{feature}'],
                   c=color,
                   edgecolors='black',
                   linewidth=1,
                   marker='s',
                   zorder=4)


def set_axes(axes):
    """
    Set the axis limits, and labels
    """

    # Set common axis features
    for ax in axes:

        # Set the x-axis limits
        ax.set_xlim(left=-10, right=110)

        # Set the y-axis
        ax.set_ylim(bottom=0, top=8)
        ax.set_ylabel('Elevation (m NAVD88)',
                      color='black',
                      fontname='Arial',
                      fontsize=12,
                      fontweight='bold')

    # Label the x-axis
    axes[-1].set_xlabel('Cross-Shore Distance (m)',
                        color='black',
                        fontname='Arial',
                        fontsize=12,
                        fontweight='bold')


def style_and_save(figure, axes):
    """
    - Set the spine style
    - Make a tight and transparent background
    - Save and close
    """

    # Set the spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

    # Make transparent and tight
    plt.tight_layout()
    figure.patch.set_color('white')
    figure.patch.set_alpha(0.0)

    # Save and close the figure
    title = 'Figure 6'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


def plot_profiles(fp, fm, np, nm):
    """
    Plot a fenced and natural profile with their
    morphometrics marked off

    - fp: Fenced profile (Dataframe)
    - fm: Fenced morphometrics (Dataframe)
    - np: Natural profile (Dataframe)
    - nm: Natural morphometrics (Dataframe)
    """

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=300)
    features = ['mhw', 'toe', 'crest', 'heel', 'fence',
                'fence_crest', 'fence_heel']
    colors = ['goldenrod', 'blue', 'magenta', 'green',
              'red', (0.91, 0.41, 0.17), (0.50, 0.00, 0.90)]
    ab = ['A.', 'B.']

    # Add a grid
    for ax in [ax1, ax2]:
        ax.grid(color='grey', linewidth=0.25, zorder=0)

    # Make the plots. This loop handles features
    # that are common to both subplots
    for ax, profile, morpho, letter in zip((ax1, ax2), [np, fp], [nm, fm], ab):

        # Plot the water
        add_water(ax=ax, x=profile['X'])

        # Plot the profile
        add_profile(ax=ax, x=profile['X'], y=profile['Y'])

        # Plot the natural dune features
        plot_features(ax=ax, morpho=morpho, features=features[:4], colors=colors[:4])

        # Label the subplot as "A" or "B"
        ax.text(x=100, y=7, s=letter, color='black',
                fontname='Arial',
                fontsize=14,
                fontweight='bold')

    # Plot fenced features
    plot_features(ax=ax2, morpho=fm, features=features[4:], colors=colors[4:])

    # Add text labels
    add_labels(ax=ax2, features=features, colors=colors)

    # Set the axes
    set_axes(axes=[ax1, ax2])

    # Style and save the figure
    style_and_save(figure=fig, axes=[ax1, ax2])


def main():
    """
    Main program function
    """

    # Load the data
    findex = 1500 + 115
    nindex = 38454 - 980 - 1500 + 597
    f_profile, f_morpho = load_data(p_type='Fenced', index=findex)
    n_profile, n_morpho = load_data(p_type='Natural', index=nindex)

    # Make the figure
    plot_profiles(fp=f_profile, fm=f_morpho, np=n_profile, nm=n_morpho)


if __name__ == '__main__':
    main()