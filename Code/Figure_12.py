#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 25, 2019

Figure 12. Distribution of beach widths on Bogue Banks. (A) Overall beach widths for Fort Macon, non-fenced areas,
and fenced areas. B) Beach width variations in Fort Macon colored by year. C) Beach width variations in non-fence
locations colored by year. D) Beach width variations in fenced locations colored by year.

- Creates the figure and a .txt file with the statistics used

@author: michaelitzkin
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import os

pd.set_option('mode.chained_assignment', None)


"""
Functions to load and format the data
"""


def chunks(l, n):
    """
    Split list l into n sized chunks

    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_data(year, variable):
    """
    Load the data into Pandas dataframes. Supply the years needed
    as the input argument

    - year: Int for the year to look at
    - variable: String with the name of the column to use
    """

    # Set a path to the data
    fname = os.path.join('..', 'Data', 'Morphometrics for Bogue ' + str(year) + '.csv')

    # Load the data and pull the needed column
    if variable is not None:
        data = pd.read_csv(fname, header=0, delimiter=',')[variable]
    else:
        data = pd.read_csv(fname, header=0, delimiter=',')

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


def beach_width_histogram_overall_subfunction(data, ax, color='black'):
    """
    Make a histogram of all beach slopes recorded on Bogue Banks
    colored by the fence density level. This is the subfunction
    because it is a version that is called by beach_width_comparison_plot()
    """

    # Melt the dataframe
    data = pd.melt(data,
                   id_vars=['Levels'],
                   value_vars=['2010', '2011', '2014', '2016'])

    # Setup the figure
    xtext, ytext, spacing = 070.00, 00.0475, 0.0025
    x_min, x_max = 0, 150
    y_min, y_max = 0, 0.05
    labels = ['Fort Macon', 'Non-Fenced', 'Fenced']
    colors = ['darkred', 'dimgray', 'cornflowerblue']

    # Plot the data
    for ii in range(0, 3):
        sns.kdeplot(data=data['value'].loc[data['Levels'] == ii],
                    shade=True,
                    color=colors[ii],
                    linewidth=2,
                    legend=False,
                    ax=ax,
                    zorder=3 + ii)

        # Add text labels
        ax.text(x=xtext,
                y=ytext - (ii * spacing),
                s=labels[ii],
                color=colors[ii],
                fontname='Arial',
                fontsize=12,
                fontweight='bold',
                zorder=3)

    # Set the x-axis
    ax.set_xlim(left=x_min, right=x_max)
    ax.set_xlabel('Beach Width (m)',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')

    # Set the y-axis
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_ylabel('Density',
                  color=color,
                  fontname='Arial',
                  fontsize=12,
                  fontweight='bold')


def beach_width_histogram_by_level_subfunction(data, axs, color='black'):
    """
    Make a 4x1 plot of beach slope where each subplot represents a level
    of sand fencing and the colors in the plot represent each survey year
    from 2010 to 2016. This is the subfunction because it is a version that
    is called by beach_width_comparison_plot()
    """

    xtext, ytext, spacing = 120, 0.050, 0.0115
    xmin, xmax = 0, 150
    ymin, ymax = 0, 0.06
    lwidth = 2
    years = ['2010', '2011', '2014', '2016']
    labels = ['Fort Macon', 'Non-Fenced', 'Fenced']
    colors = ['lightsalmon', 'tomato', 'r', 'brown']

    # Loop through the axes and years then plot
    for level, ax in zip(range(0, 3), axs):
        for col, year in enumerate(years):

            sns.kdeplot(data=data[year].loc[data['Levels'] == level],
                        shade=True,
                        color=colors[col],
                        linewidth=lwidth,
                        legend=False,
                        ax=ax,
                        zorder=3 + level)

        # Set the x-axis
        ax.set_xlim(left=xmin, right=xmax)

        # Set the y-axis
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_yticks([0, 0.03, 0.06])
        ax.set_ylabel(labels[level],
                      color=color,
                      fontname='Arial',
                      fontsize=12,
                      fontweight='bold')

        axs[level].tick_params(colors=color)

    # Set the x-axis
    axs[2].set_xlabel('Beach Width (m)',
                      color=color,
                      fontname='Arial',
                      fontsize=12,
                      fontweight='bold')

    for level in range(0, 4):
        # Add text labels to the first axis
        axs[0].text(x=xtext,
                    y=ytext - (level * spacing),
                    s=years[level],
                    color=colors[level],
                    fontname='Arial',
                    fontsize=12,
                    fontweight='bold',
                    zorder=3)


def beach_width_comparison_plot(data, color='black'):
    """
    Combine beach_width_histogram_overal() and beach_width_histogram_by_level()
    into a single function call
    """

    # Setup the figure
    fig = plt.figure(dpi=300)
    ax1 = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    ax4 = plt.subplot2grid((3, 4), (2, 2), colspan=2)

    # Add grids
    add_grid([ax1, ax2, ax3, ax4], xax=True)

    # Plot the overall data
    beach_width_histogram_overall_subfunction(data, ax1, color=color)

    # Plot the yearly data
    beach_width_histogram_by_level_subfunction(data, [ax2, ax3, ax4], color=color)

    # Add letter labels
    whys = [0.0475, 0.05, 0.05, 0.05]
    letters = ['A.', 'B.', 'C.', 'D.']
    for ax, yy, letter in zip([ax1, ax2, ax3, ax4], whys, letters):
        ax.text(x=5, y=yy, s=letter, color='black',
                fontname='Arial',
                fontsize=14,
                fontweight='bold',
                zorder=5)

    # Set spines
    set_spines([ax1, ax2, ax3, ax4], color=color)

    # Make the figure tight and transparent
    tight_and_transparent(fig, [ax1, ax2, ax3, ax4], color=color,
                          w_pad=1)

    # Save the figure
    title = 'Figure 12'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


"""
Functions to make a .txt file with stats
"""


def write_header(f, n, l):
    """
    Add a header to a section of a text file

    - f: open text file
    - n: name of section
    - l: length of header

    Called by:
    - interannual_beach_width_stats()
    """

    header_len = l
    section_part = n
    lead_in = '#' * 2
    end_out = '#' * (header_len - len(section_part) - 4)
    f.write('%s\n' % ('#' * header_len))
    f.write('%s %s %s\n' % (lead_in, section_part, end_out))
    f.write('%s\n\n' % ('#' * header_len))


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


def summary_statistics(data, decimals):
    """
    Compute summary statistics for natural dune morphology

    data: Data to work on
    year: String with the year(s) of the current data
    decimals: Decimal places to round to when writing to the text file

    Called by:
    - natural_dune_morphology_aggregate_stats()
    """

    # data = np.log(data)

    # Calculate the mean
    average = np.nanmean(data)

    # Calculate 2 times the standard deviation
    sd2 = np.nanstd(data)

    # Calculate the median
    middle = np.nanmedian(data)

    # Calculate the mode
    most = stats.mode(data)[0]

    # Calculate the 95% confidence interval
    lo, hi, _ = confidence_intervals(data)

    # Calculate the 25, 50, and 75 percentile
    q25 = np.nanpercentile(data, 25)
    q50 = np.nanpercentile(data, 50)
    q75 = np.nanpercentile(data, 75)

    # Calculate the skewness and kurtosis
    skew = stats.skew(data, nan_policy='omit')
    kurt = stats.kurtosis(data, nan_policy='omit')

    # Calculate the standard error on the mean
    # autocorr = estimated_autocorrelation(data)
    # sig_vals = np.where(autocorr <= 0.05)
    # n_star = np.nanmin(sig_vals)
    # sem = sd2 / np.sqrt(n_star)
    sem = stats.sem(data, nan_policy='omit')

    # Write the data in a string
    res = '\tMean: %s\tSD: %s \tMedian: %s\tMode: %s\t95 Pct CI: [%s %s] (%s)\tIQR: [%s %s %s] (%s)\tSkewness: %s\tKurtosis: %s\tSEM: %s\r\n' %\
          (str(np.around(average, decimals=decimals)),
           str(np.around(sd2, decimals=decimals)),
           str(np.around(middle, decimals=decimals)),
           str(np.around(most[0], decimals=decimals)),
           str(np.around(lo, decimals=decimals)),
           str(np.around(hi, decimals=decimals)),
           str(np.around(hi - lo, decimals=2)),
           str(np.around(q25, decimals=decimals)),
           str(np.around(q50, decimals=decimals)),
           str(np.around(q75, decimals=decimals)),
           str(np.around(q75 - q25, decimals=decimals)),
           str(np.around(skew, decimals=decimals)),
           str(np.around(kurt, decimals=decimals)),
           str(np.around(sem, decimals=decimals)))

    return res


def interannual_beach_width_stats(data, alpha=0.05):
    """
    Calculate statistics pertaining to the interannual variations
    in beach width along Bogue Banks from 2010-2016

    Called by:
    - Beach_Width_Comparison.py
    """

    # Set the header length
    header_length = 50

    # Set a list of years and locations
    years = ['2010', '2011', '2014', '2016']
    locations = ['Fort Macon', 'Non-Fenced', 'Fenced\t']

    # Melt the data so that all years are in one column and all levels are in another
    data = pd.melt(data, id_vars='Levels', value_vars=['2010', '2011', '2014', '2016'])
    data.columns = ['Location', 'Year', 'Value']

    # Create a .txt file
    partial_fname = '(Figure 12) Interannual Beach Width Statistics.txt'
    fname = os.path.join('..', 'Text Files', partial_fname)
    f = open(fname, 'w')

    # Write a header
    f.write('Interannual Beach Width Statistics for Bogue Banks \r\n')
    f.write('\t- Summary statistics for data aggregated by location\r\n')
    f.write('\t- Kruskal-Wallis H-Tests to compare Fort Macon, non-fenced, and fenced locations\r\n')
    f.write('\t- Mann-Whitney U-Test with Bonferroni to further compare locations in aggregate\r\n')
    f.write('\t- Kruskal-Wallis H-Tests to compare Fort Macon, non-fenced, and fenced locations by year\r\n')
    f.write('\t- Mann-Whitney U-Test with Bonferroni to further compare locations by year\r\n\r\n\r\n')

    # Write summary statistics for the aggregate data
    write_header(f, 'Summary Statistics (Aggregate)', header_length)
    for ii, location in enumerate(locations):
        f.write('%s:%s' % (location, summary_statistics(data['Value'].loc[data['Location'] == ii], 1)))
    f.write('\r\n\r\n')

    # Perform a Kruskal-Wallis H-test
    write_header(f, 'Compare Beach Widths across all locations', header_length)
    f.write('Kruskal-Wallis H-test (alpha = %0.2f)\r\n' % alpha)
    f.write('H0: The population median of all groups are equal\r\n')
    f.write('H1: The population median of all groups are not equal\r\n\r\n')
    h, p = stats.kruskal(data['Value'].loc[data['Location'] == 0],
                         data['Value'].loc[data['Location'] == 1],
                         data['Value'].loc[data['Location'] == 2], nan_policy='omit')
    if p < alpha:
        f.write('Reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (h, p))
    else:
        f.write('Fail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (h, p))
    f.write('\r\n\r\n')

    # Perform unequal variance T-tests to compare locations to each other
    write_header(f, 'Compare locations against each other', header_length)
    f.write('Mann-Whitney U-test (Bonferroni alpha = %0.2f)\r\n' % (alpha/3))
    f.write('H0: The sample populations have the same distribution\r\n')
    f.write('H1: The samples populations do not have the same distribution\r\n\r\n')

    u1, p1 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0], y=data['Value'].loc[data['Location'] == 1],
                                alternative='two-sided')
    u2, p2 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0], y=data['Value'].loc[data['Location'] == 2],
                                alternative='two-sided')
    u3, p3 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 1], y=data['Value'].loc[data['Location'] == 2],
                                alternative='two-sided')

    if p1 < alpha/3:
        f.write('Fort Macon v. Non-Fenced: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
    else:
        f.write('Fort Macon v. Non-Fenced: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
    if p2 < alpha:
        f.write('Fort Macon v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
    else:
        f.write(
            'Fort Macon v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
    if p3 < alpha:
        f.write('Non-Fenced v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
    else:
        f.write('Non-Fenced v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
    f.write('\r\n\r\n')

    # Write summary statistics by year for the data
    write_header(f, 'Summary Statistics (By Year)', header_length)
    for yr in years:
        f.write('\r\n%s:\r\n' % yr)
        for ii, location in enumerate(locations):
            stats_string = summary_statistics(data['Value'].loc[data['Location'] == ii].loc[data['Year'] == yr], 1)
            f.write('%s:%s' % (location, stats_string))
    f.write('\r\n\r\n')

    # Perform Kruskal-Wallis to compare a single location
    write_header(f, 'Compare a location for all years', header_length)
    f.write('Kruskal-Wallis H-test comparing all years for a single location (alpha = %0.2f)\r\n' % alpha)
    f.write('H0: The population median of all years are equal\r\n')
    f.write('H1: The population median of all years are not equal\r\n\r\n')
    for ii, loc in enumerate(locations):
        h, p = stats.kruskal(data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2010'],
                             data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2011'],
                             data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2014'],
                             data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2016'], nan_policy='omit')
        if p < alpha:
            f.write('%s:\tReject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))
        else:
            f.write('%s:\tFail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))
    f.write('\r\n\r\n')

    # Compare consecutive survey years for a given location
    write_header(f, 'Compare years for a single location', header_length)
    f.write('Mann-Whitney U-test comparing years for a given location (Bonferroni alpha = %0.2f)\r\n' % (alpha/3))
    f.write('H0: The sample populations have the same distribution\r\n')
    f.write('H1: The samples populations do not have the same distribution\r\n\r\n')
    for ii, loc in enumerate(locations):
        f.write('\r\n%s:\r\n' % loc)
        u1, p1 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2010'],
                                    y=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2011'],
                                    alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2011'],
                                    y=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2014'],
                                    alternative='two-sided')
        u3, p3 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2014'],
                                    y=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2016'],
                                    alternative='two-sided')
        if p1 < alpha/3:
            f.write('2010 v. 2011:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
        else:
            f.write('2010 v. 2011:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
        if p2 < alpha/3:
            f.write('2011 v. 2014:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
        else:
            f.write('2011 v. 2014:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
        if p3 < alpha/3:
            f.write('2014 v. 2016:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
        else:
            f.write('2014 v. 2016:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))

    # Close the file
    f.close()


"""
Run the program
"""


def main():
    """
    Main program function
    """

    # Load the data
    data_2010 = load_data(year=2010, variable=None)
    data_2011 = load_data(year=2011, variable=None)
    data_2014 = load_data(year=2014, variable=None)
    data_2016 = load_data(year=2016, variable=None)

    # Set a profile cutoff for Fort Macon
    fort_macon = 2000

    # Add a "category" column to all the dataframes
    data_2010['Category'] = np.nan
    data_2011['Category'] = np.nan
    data_2014['Category'] = np.nan
    data_2016['Category'] = np.nan

    # Set all the Fort Macon profiles to 2
    data_2010['Category'][-fort_macon:] = 0
    data_2011['Category'][-fort_macon:] = 0
    data_2014['Category'][-fort_macon:] = 0
    data_2016['Category'][-fort_macon:] = 0

    # Set non-Fort Macon profiles based on fences. Fences are the same for all years so only
    # need to compare to 2010
    for row in range(fort_macon):
        if data_2010['x_fence'][row] >= 0:
            data_2010['Category'][row] = 2
            data_2011['Category'][row] = 2
            data_2014['Category'][row] = 2
            data_2016['Category'][row] = 2
        else:
            data_2010['Category'][row] = 1
            data_2011['Category'][row] = 1
            data_2014['Category'][row] = 1
            data_2016['Category'][row] = 1

    data = pd.concat([data_2010['Beach Width'],
                      data_2011['Beach Width'],
                      data_2014['Beach Width'],
                      data_2016['Beach Width']],
                     axis=1)
    data.columns = ['2010', '2011', '2014', '2016']

    # Assign a category to each dataframe
    data['Levels'] = data_2010['Category']

    # Make figures
    beach_width_comparison_plot(data, color='black')

    # Calculate statistics
    interannual_beach_width_stats(data)


if __name__ == '__main__':
    main()