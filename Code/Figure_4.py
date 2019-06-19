#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 19, 2019

Figure 4: Timeline of storms that have impacted Carteret (Bogue Banks),
and the adjacent counties to the north (Dare) and south (Onslow) between
1997 and 2016. Data from Sefcovic (2016)

Sefcovic, Z.P. (2016). Climatology of Tropical Cyclones in Eastern
North Carolina. Retrieved from
https://www.weather.gov/mhx/TropicalClimatology

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def load_and_group_storms():
    """
    Load the NC storm record from Sefcovic (2016)
    and count the storm categories by year
    """

    # Load the data
    fname = os.path.join('..', 'Data', 'NC Storm Record.csv')
    data = pd.read_csv(fname, header=0)

    # Remove years before 1997 from the record
    data = data.loc[data['Year'] >= 1997]

    # Set a list of years with no storms
    no_storms = [1997, 2000, 2009]

    # Loop through the years and count the amount of storms
    # for each category and place them in a dictionary of
    # storm counts
    counts = dict()
    for year in range(1997, 2017):

        if year in no_storms:
            counts[str(year)] = [0, 0, 0]

        else:
            curr_counts = [
                len(data['Year'].loc[(data['Year'] == year) & (data['Category'] == 0)]),
                len(data['Year'].loc[(data['Year'] == year) & (data['Category'] == 1)]),
                len(data['Year'].loc[(data['Year'] == year) & (data['Category'] == 2)])
            ]

            counts[str(year)] = curr_counts

    # Convert counts to a DataFrame
    counts = pd.DataFrame(counts)

    # Create a row for the total storm counts
    counts.loc['Total'] = pd.Series(counts.sum())
    counts.loc['Cat 0-1'] = pd.Series(counts.loc[0, :] + counts.loc[1, :])

    # Reshape the table
    counts = counts.transpose()

    return counts


def plot_storms(df):
    """
    Make a barplot of storm occurences over the
    study period with each year broken down by
    storm category.

    df: DataFrame with storm counts
    """

    # Setup the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    colors = ['goldenrod', 'darkorange', 'darkblue']
    columns = ['Total', 'Cat 0-1', '0']

    # Add a grid
    ax.grid(color='grey', linewidth=0.25, zorder=0)

    # Annotate the bar for 2010, noting when
    # sand fences were installed
    ax.annotate('Sand Fences Installed',
                xy=(13, 1.0),
                xytext=(13, 3.25),
                horizontalalignment='left',
                rotation=45,
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontname='Arial',
                fontsize=12,
                fontweight='normal',
                zorder=3)

    # Plot the data
    for counter, column in enumerate(columns):
        if column == '0':
            column = 0
        sns.barplot(x=df.index,
                    y=column,
                    data=df,
                    color=colors[counter],
                    edgecolor='black',
                    linewidth=1,
                    ax=ax,
                    zorder=3)

    # Add text labels
    x_text, y_text, spacing = 0, 3.70, 0.25
    labels = ['Tropical: 20', 'Category 1: 4', 'Category 2: 5']
    for counter, label in enumerate(labels):
        ax.text(x=x_text,
                y=y_text - (counter * spacing),
                s=label,
                color=colors[2 - counter],
                fontname='Arial',
                fontsize=14,
                fontweight='bold',
                zorder=3)

    # Set the x-axis
    plt.xticks(rotation='vertical')

    # Set the y-axis
    ax.set_ylim(bottom=0, top=4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel('No. Storms',
                  color='black',
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Set a tight layout and transparent background
    [tick.set_fontname('Arial') for tick in ax.get_xticklabels()]
    [tick.set_fontname('Arial') for tick in ax.get_yticklabels()]
    plt.tight_layout()
    fig.patch.set_color('white')
    fig.patch.set_alpha(0.0)

    # Save and close the figure
    title = 'Figure 4'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


def main():
    """
    Main program function
    """

    # Load the storm record and group the
    # storms by year and category
    counts = load_and_group_storms()

    # Make the figure
    plot_storms(df=counts)


if __name__ == '__main__':
    main()