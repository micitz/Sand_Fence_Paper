#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 25, 2019

Figure 11. Temporal variation of natural dune height, width, and volume for natural dunes in Fort Macon,
non-fenced areas, and fenced areas.

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
Functions to load and process data
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


def make_natural_morphology_boxplot(data, color='black'):
    """
    Make a three panel boxplot comapring natural dune morphology across
    years and separated by alongshore fence density
    """

    # Setup the plot
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, dpi=300, sharex=True)
    types = ['Height\n(m NAVD88)', 'Width\n(m)', 'Volume\n($m^{3}/m$)']
    colors = ['darkred', 'dimgray', 'cornflowerblue']
    ymin, ymax = 0, [10, 85, 175]

    # Add a grid
    add_grid([ax1, ax2, ax3], xax=False)

    # Loop through and plot
    for i, ax in enumerate((ax1, ax2, ax3)):

        data[i] = pd.melt(data[i],
                          id_vars=['Levels'],
                          value_vars=['2010', '2011', '2014', '2016'])

        sns.boxplot(x='variable',
                    y='value',
                    hue='Levels',
                    data=data[i],
                    palette=colors,
                    linewidth=1.25,
                    ax=ax,
                    zorder=3 + i)

        # Set the x-axis
        if i != 2:
            ax.set_xlabel('')
            ax.tick_params(bottom=False)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('')

        # Set the y-axis
        ax.set_ylabel(types[i],
                      color=color,
                      fontname='Arial',
                      fontsize=12,
                      fontweight='bold')

        # Set general axis parameters
        ax.set_ylim(bottom=ymin, top=ymax[i])
        ax.legend_.remove()

    # Set the spines
    set_spines([ax1, ax2, ax3], color)

    # Add a color coded key above the top plot
    xs = [0.2, 0.5, 0.75]
    labels = ['Fort Macon', 'Non-Fenced', '    Fenced']
    for ii, label in enumerate(labels):
        fig.text(x=xs[ii],
                 y=0.97,
                 s=label,
                 color=colors[ii],
                 fontname='Arial',
                 fontsize=12,
                 fontweight='bold')

    # Set a tight layout and a transparent background
    tight_and_transparent(fig, [ax1, ax2, ax3], color)

    # Save the figure
    title = 'Figure 11'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


"""
Functions to make .txt files with stats
"""


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


def natural_dune_morphology_nonaggregate_stats(data, param, legacy, alpha=0.05, dec=2):
    """
    Compare natural dune morphology for Fort Macon, non-fenced, and
    fenced locations. This compares the data in aggregate, so the 2010,
    2011, 2014, and 2016 data is combined here.

    - data: dataframe with data to work on
    - param: string with the metric being analyzed
    - alpha: confidence level to test (default = 0.05)
    - dec: number of decimals to round to in the stats file

    Called by:
    - Natural_Dune_Morphology_Comparison.py
    """

    if legacy:

        # Set a list of years and locations
        years = ['1997', '2010', '2011', '2014', '2016']
        locations = ['Fort Macon', 'Non-Fenced', 'Fenced\t']

        # Melt the data so that all years are in one column and all levels are in another
        data = pd.melt(data, id_vars='Levels', value_vars=['1997', '2010', '2011', '2014', '2016'])
        data.columns = ['Location', 'Year', 'Value']

        # Open a text file to write to
        fname = os.path.join('..', 'Text Files', 'Natural Dune ' + param + ' Comparison by Year.txt')
        f = open(fname, 'w')

        # Write a header to the file
        f.write('Compare natural dune %s along Bogue Banks by location (all years combined)\n' % param)
        f.write('\t- Summary statistics of the data\r\n')
        f.write('\t- Test if the data is normally distributed (alpha = %0.2f)\r\n' % alpha)
        f.write('\t- One-Way ANOVA to compare means, if normal (alpha = %0.2f)\r\n' % alpha)
        f.write('\t- Kruskal-Wallis to compare medians, if not normal (alpha = %0.2f)\r\n' % alpha)
        f.write('\t- Mann-Whitney U-tests to compare populations to each other with Bonferroni Correction (Bonferroni alpha = %0.2f)\r\n' % (alpha/3))
        f.write('\r\n')

        # Write summary statistics
        f.write('\r\n##############################\r\n')
        f.write('Summary Statistics:\r\n')
        for yr in years:
            f.write('\r\n%s:\r\n' % yr)
            for ii, location in enumerate(locations):
                f.write('%s%s' % (location, summary_statistics(data['Value'].loc[(data['Location'] == ii) & (data['Year'] == yr)], decimals=dec)))

        # Perform a Kruskal-Wallis H-test
        f.write('\r\n##############################\r\n')
        f.write('Kruskal-Wallis H-test comparing all locations for a single year (alpha = %0.2f)\r\n' % alpha)
        f.write('H0: The population median of all locations are equal\r\n')
        f.write('H1: The population median of all locations are not equal\r\n\r\n')
        for yr in years:
            h, p = stats.kruskal(data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                 data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                 data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr], nan_policy='omit')
            if p < alpha:
                f.write('%s:\tReject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (yr, h, p))
            else:
                f.write('%s:\tFail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (yr, h, p))

        # Perform a Kruskal-Wallis H-test
        f.write('\r\n##############################\r\n')
        f.write('Kruskal-Wallis H-test comparing all years for a single location (alpha = %0.2f)\r\n' % alpha)
        f.write('H0: The population median of all years are equal\r\n')
        f.write('H1: The population median of all years are not equal\r\n\r\n')
        for ii, loc in enumerate(locations):
            h, p = stats.kruskal(data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '1997'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2010'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2011'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2014'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2016'], nan_policy='omit')
            if p < alpha:
                f.write('%s:\tReject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))
            else:
                f.write('%s:\tFail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))

        # Perform unequal variances T-tests to compare groups to each other
        f.write('\r\n##############################\r\n')
        f.write('Mann-Whitney U-test (Bonferroni alpha = %0.2f)\r\n' % (alpha/3))
        f.write('H0: The sample populations have the same distribution\r\n')
        f.write('H1: The samples populations do not have the same distribution\r\n\r\n')
        for yr in years:
            f.write('\r\n%s:\r\n' % yr)
            u1, p1 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                        alternative='two-sided')
            u2, p2 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr],
                                        alternative='two-sided')
            u3, p3 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr],
                                        alternative='two-sided')
            if p1 < alpha/3:
                f.write('Fort Macon v. Non-Fenced: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            else:
                f.write(
                    'Fort Macon v. Non-Fenced: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            if p2 < alpha/3:
                f.write('Fort Macon v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            else:
                f.write(
                    'Fort Macon v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            if p3 < alpha/3:
                f.write('Non-Fenced v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
            else:
                f.write('Non-Fenced v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))

        # Perform unequal variances T-tests to compare groups to each other
        f.write('\r\n##############################\r\n')
        f.write('Mann-Whitney U-test (Bonferroni alpha = %0.2f)\r\n' % (alpha/4))
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
            u4, p4 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '1997'],
                                        y=data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2010'],
                                        alternative='two-sided')
            if p4 < alpha/4:
                f.write('1997 v. 2010:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u4, p4))
            else:
                f.write('1997 v. 2010:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u4, p4))
            if p1 < alpha/4:
                f.write('2010 v. 2011:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            else:
                f.write('2010 v. 2011:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            if p2 < alpha/4:
                f.write('2011 v. 2014:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            else:
                f.write('2011 v. 2014:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            if p3 < alpha/4:
                f.write('2014 v. 2016:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
            else:
                f.write('2014 v. 2016:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))

        # Close the file
        f.close()
        print(f'Text file saved: {fname}')

    else:

        # Set a list of years and locations
        years = ['2010', '2011', '2014', '2016']
        locations = ['Fort Macon', 'Non-Fenced', 'Fenced\t']

        # Melt the data so that all years are in one column and all levels are in another
        data = pd.melt(data, id_vars='Levels', value_vars=['2010', '2011', '2014', '2016'])
        data.columns = ['Location', 'Year', 'Value']

        # Open a text file to write to
        fname = os.path.join('..', 'Text Files', f'(Figure 11) Natural Dune {param} Comparison by Year.txt')
        f = open(fname, 'w')

        # Write a header to the file
        f.write('Compare natural dune %s along Bogue Banks by location (all years combined)\n' % param)
        f.write('\t- Summary statistics of the data\r\n')
        f.write('\t- Test if the data is normally distributed (alpha = %0.2f)\r\n' % alpha)
        f.write('\t- One-Way ANOVA to compare means, if normal (alpha = %0.2f)\r\n' % alpha)
        f.write('\t- Kruskal-Wallis to compare medians, if not normal (alpha = %0.2f)\r\n' % alpha)
        f.write(
            '\t- Mann-Whitney U-tests to compare populations to each other with Bonferroni Correction (Bonferroni alpha = %0.2f)\r\n' % (
                        alpha / 3))
        f.write('\r\n')

        # Write summary statistics
        f.write('\r\n##############################\r\n')
        f.write('Summary Statistics:\r\n')
        for yr in years:
            f.write('\r\n%s:\r\n' % yr)
            for ii, location in enumerate(locations):
                f.write('%s%s' % (location,
                                  summary_statistics(data['Value'].loc[(data['Location'] == ii) & (data['Year'] == yr)],
                                                     decimals=dec)))

        # Perform a Kruskal-Wallis H-test
        f.write('\r\n##############################\r\n')
        f.write('Kruskal-Wallis H-test comparing all locations for a single year (alpha = %0.2f)\r\n' % alpha)
        f.write('H0: The population median of all locations are equal\r\n')
        f.write('H1: The population median of all locations are not equal\r\n\r\n')
        for yr in years:
            h, p = stats.kruskal(data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                 data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                 data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr], nan_policy='omit')
            if p < alpha:
                f.write('%s:\tReject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (yr, h, p))
            else:
                f.write('%s:\tFail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (yr, h, p))

        # Perform a Kruskal-Wallis H-test
        f.write('\r\n##############################\r\n')
        f.write('Kruskal-Wallis H-test comparing all years for a single location (alpha = %0.2f)\r\n' % alpha)
        f.write('H0: The population median of all years are equal\r\n')
        f.write('H1: The population median of all years are not equal\r\n\r\n')
        for ii, loc in enumerate(locations):
            h, p = stats.kruskal(data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2010'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2011'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2014'],
                                 data['Value'].loc[data['Location'] == ii].loc[data['Year'] == '2016'],
                                 nan_policy='omit')
            if p < alpha:
                f.write('%s:\tReject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))
            else:
                f.write('%s:\tFail to reject the null hypothesis (H = %0.4f\tp = %0.4f)\r\n' % (loc, h, p))

        # Perform unequal variances T-tests to compare groups to each other
        f.write('\r\n##############################\r\n')
        f.write('Mann-Whitney U-test (Bonferroni alpha = %0.2f)\r\n' % (alpha / 3))
        f.write('H0: The sample populations have the same distribution\r\n')
        f.write('H1: The samples populations do not have the same distribution\r\n\r\n')
        for yr in years:
            f.write('\r\n%s:\r\n' % yr)
            u1, p1 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                        alternative='two-sided')
            u2, p2 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 0].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr],
                                        alternative='two-sided')
            u3, p3 = stats.mannwhitneyu(x=data['Value'].loc[data['Location'] == 1].loc[data['Year'] == yr],
                                        y=data['Value'].loc[data['Location'] == 2].loc[data['Year'] == yr],
                                        alternative='two-sided')
            if p1 < alpha / 3:
                f.write('Fort Macon v. Non-Fenced: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            else:
                f.write(
                    'Fort Macon v. Non-Fenced: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (
                    u1, p1))
            if p2 < alpha / 3:
                f.write('Fort Macon v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            else:
                f.write(
                    'Fort Macon v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            if p3 < alpha / 3:
                f.write('Non-Fenced v. Fenced\t: Reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
            else:
                f.write(
                    'Non-Fenced v. Fenced\t: Fail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))

        # Perform unequal variances T-tests to compare groups to each other
        f.write('\r\n##############################\r\n')
        f.write('Mann-Whitney U-test (Bonferroni alpha = %0.2f)\r\n' % (alpha / 4))
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

            if p1 < alpha / 4:
                f.write('2010 v. 2011:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            else:
                f.write('2010 v. 2011:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
            if p2 < alpha / 4:
                f.write('2011 v. 2014:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            else:
                f.write('2011 v. 2014:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
            if p3 < alpha / 4:
                f.write('2014 v. 2016:\tReject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))
            else:
                f.write('2014 v. 2016:\tFail to reject the null hypothesis (U = %0.4f\tp = %0.4f)\r\n' % (u3, p3))

        # Close the file
        f.close()
        print(f'Text file saved: {fname}')


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
    fort_macon = 2229

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

    # Set fenced profiles
    data_2010['Category'][:fort_macon].loc[data_2010['x_fence'] > 0] = 2
    data_2011['Category'][:fort_macon].loc[data_2011['x_fence'] > 0] = 2
    data_2014['Category'][:fort_macon].loc[data_2014['x_fence'] > 0] = 2
    data_2016['Category'][:fort_macon].loc[data_2016['x_fence'] > 0] = 2

    data_2010['Category'][:fort_macon].loc[(data_2010['Category'] != 0) & (data_2010['Category'] != 2)] = 1
    data_2011['Category'][:fort_macon].loc[(data_2011['Category'] != 0) & (data_2011['Category'] != 2)] = 1
    data_2014['Category'][:fort_macon].loc[(data_2014['Category'] != 0) & (data_2014['Category'] != 2)] = 1
    data_2016['Category'][:fort_macon].loc[(data_2016['Category'] != 0) & (data_2016['Category'] != 2)] = 1

    # Sort out fences using chunks
    for data in [data_2010, data_2011, data_2014, data_2016]:
        for n in chunks(data[:fort_macon], 5):
            if np.any(data['x_fence'].iloc[n.index] > 0):
                data['Category'].iloc[n.index] = 2
            else:
                data['Category'].iloc[n.index] = 1

    # Make dhigh, dwide, and dvol dataframes
    dhigh_col = 'y_crest'
    dvol_col = 'Natural Dune Volume'
    dhigh = pd.concat([
        data_2010[dhigh_col],
        data_2011[dhigh_col], data_2014[dhigh_col],
        data_2016[dhigh_col]],
        axis=1)
    dhigh.columns = ['2010', '2011', '2014', '2016']
    dvol = pd.concat(
        [data_2010[dvol_col], data_2011[dvol_col],
         data_2014[dvol_col], data_2016[dvol_col]],
        axis=1)
    dvol.columns = ['2010', '2011', '2014', '2016']
    dwide = pd.DataFrame({
        '2010': data_2010['x_heel'] - data_2010['x_toe'],
        '2011': data_2011['x_heel'] - data_2011['x_toe'],
        '2014': data_2014['x_heel'] - data_2014['x_toe'],
        '2016': data_2016['x_heel'] - data_2016['x_toe']
    })

    # Make a beach width data frame
    bwidth = pd.concat(
        [data_2010['Beach Width'], data_2011['Beach Width'], data_2014['Beach Width'],
         data_2016['Beach Width']],
        axis=1)
    bwidth.columns = ['2010', '2011', '2014', '2016']

    # Assign a category to each dataframe
    dhigh['Levels'] = data_2010['Category']
    dwide['Levels'] = data_2010['Category']
    dvol['Levels'] = data_2010['Category']
    bwidth['Levels'] = data_2010['Category']

    # Any profile with a width or volume <= 0 had a measurment error. Remove those. Same for colossal
    # beach width changes
    years = ['2010', '2011', '2014', '2016']
    dhigh[years] = dhigh[years][dhigh[years] > 0]
    dwide[years] = dwide[years][dwide[years] > 0]
    dvol[years] = dvol[years][dvol[years] > 0]

    # Make the figure
    make_natural_morphology_boxplot([dhigh, dwide, dvol], color='black')

    # Perform statistical tests comparing the data across locations but grouped by year
    natural_dune_morphology_nonaggregate_stats(dhigh.dropna(), 'Height', legacy=False, alpha=0.05, dec=1)
    natural_dune_morphology_nonaggregate_stats(dwide.dropna(), 'Width', legacy=False, alpha=0.05, dec=1)
    natural_dune_morphology_nonaggregate_stats(dvol.dropna(), 'Volume', legacy=False, alpha=0.05, dec=1)


if __name__ == '__main__':
    main()
