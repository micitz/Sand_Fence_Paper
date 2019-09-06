#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 21, 2019

Figure 7: A) Alongshore shoreline change for Bogue Banks for 1997-2016 (green), 1997-2010 (red), and 2010-2016 (blue).
B) Density distributions of shoreline change for Bogue Banks for 1997-2016, 1997-2010, and 2010-2016.
Positive values indicate progradation, negative values indicate erosion. C) Median shoreline change for the entire
island, Fort Macon, non-fenced areas, and fenced areas colored by time period (with 95% confidence intervals).

- Creates the figure and a .txt file with the statistics used

@author: michaelitzkin
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import os

import utility_functions as utils
import plot_settings as sets
opts = sets.settings()

pd.set_option('mode.chained_assignment', None)

"""
Utility functions
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


def chunks(l, n):
    """
    Split list l into n sized chunks

    https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def dhigh_uncertainty(t0, tf, d0, df):
    """
    Calculate uncertainty in the position of Dhigh
    """

    # Load the dune positions for the current years to calculate uncertainty. Find the dune
    # slope as well from this
    data_begin = load_data(year=int(t0), variable=['x_toe', 'y_toe', 'x_crest', 'y_crest'])
    data_end = load_data(year=int(tf), variable=['x_toe', 'y_toe', 'x_crest', 'y_crest'])

    data_begin['Dune Slope'] = \
        (data_begin['y_crest'] - data_begin['y_toe']) / abs(data_begin['x_crest'] - data_begin['x_toe'])
    data_end['Dune Slope'] = \
        (data_end['y_crest'] - data_end['y_toe']) / abs(data_end['x_crest'] - data_end['x_toe'])

    # Find the uncertainty in Y as the difference between the measured y-value
    # and the average y-value. Do this for both years
    y_mean_yr1 = np.nanmean(data_begin['y_crest'].dropna())
    y_mean_yr2 = np.nanmean(data_end['y_crest'].dropna())
    y_error_yr1 = abs(data_begin['y_crest'] - y_mean_yr1)
    y_error_yr2 = abs(data_end['y_crest'] - y_mean_yr2)

    # Find the uncertainty in X as the vertical error in the LiDAR
    # data multiplied by the beach slope
    x_error_yr1 = lidar_vertical_error(t0) / data_begin['Dune Slope']
    x_error_yr2 = lidar_vertical_error(tf) / data_begin['Dune Slope']

    # Find uncertainty based on the confidence interval
    _, _, ci0 = confidence_intervals(d0)
    _, _, cif = confidence_intervals(df)

    # Find the positional uncertainty based on the uncertainty in X and Y
    up1 = np.sqrt((x_error_yr1 ** 2) + (y_error_yr1 ** 2) + (ci0 ** 2))
    up2 = np.sqrt((x_error_yr2 ** 2) + (y_error_yr2 ** 2) + (cif ** 2))

    return up1, up2


def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Called by:
    - rate_uncertainty()
    """
    n = len(x)
    variance = np.nanvar(x)
    x -= np.nanmean(x)
    r = np.correlate(x, x, mode='full')[-n:]
    # assert np.allclose(r, np.array([(x[:n - k]*x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * np.arange(n, 0, -1))
    return result


def fft_smooth(y, points):
    """
    Smooth noisy data with an FFT filter

    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    """
    box = np.ones(points)/points
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def lidar_vertical_error(yr):
    """
    Return the vertical error of the lidar data for the
    given year. Returns in meters
    """

    if yr in ['1997', '1998', '1999', '2000', '2004', '2010', '2011']:
        return 0.15
    elif yr == '2005':
        return 0.20
    elif yr == '2014':
        return 0.062
    elif yr == '2016':
        return 0.20


def mean_residual_error(df, delta):
    """
    Calculate the mean residual error for the initial and final
    time periods. Than use that to calculate an average rate error.
    - df: Dataframe being worked on
    - delta: Dataframe of change rates
    - window: Window to average over
    """

    # Perform an autocorrelation to determine the window size
    autocorr = estimated_autocorrelation(delta)
    sig_vals = np.where(autocorr <= 0.05)
    n_star = np.nanmin(sig_vals)

    row = 0
    for beta in chunks(delta, n_star):

        # Calculate the average change rate of the
        # current chunk of profiles
        mu_beta = np.nanmean(beta)

        # Calculate the residuals for the current chunk
        residuals = beta - mu_beta

        # Calculate the MRE for the current chunk
        mre = np.sqrt((1 / len(beta)) * np.nansum(residuals ** 2))

        # Add the error to the dataframe
        df[row:row+len(beta)] = mre

        # Increment the row
        row += len(beta)

    return df


def rate_uncertainty(t0, tf, pos_uncertainty):
    """
    Calculate uncertainty in the change rate, based on
    methods in Hapke et al. (2011)

    - t0: beginning time
    - tf: end time
    - pos_uncertainties: positional uncertainty

    Called by:
    - shoreline_change_df()
    """

    # Convert the times to ints
    t0, tf = int(t0), int(tf)

    # Find the rate uncertainty based on the positional uncertainties
    ur = pos_uncertainty / (tf - t0)

    return ur


def regional_uncertainty(r, ur, alpha=0.05):
    """
    Calculate reigonal uncertainty. From Hapke et al (2011),
    equation 9

    - r: Calculated rates of change
    - alpha: significance value (default = 0.05)
    """

    # Perform an autocorrelation on r to determine
    # the number of independent samples
    r_acr = estimated_autocorrelation(ur[~np.isnan(ur)])
    r_acr_vals = np.where(r_acr <= alpha)
    n_star = np.nanmin(r_acr_vals)

    # Calculate the mean uncertainty
    Ur_bar = np.nanmean(ur)

    # Calculate the uncertainty
    Urq_bar = (1 / np.sqrt(n_star)) * Ur_bar
    Urq_bar = np.around(Urq_bar, decimals=2)
    return Urq_bar


"""
Functions to load an format data
"""


def change_df(d1997, d2010, d2016):
    """
    Make a dataframe of change rates between lidar
    surveys and their smoothed trends

    - d1997: Dataframe with values from 1997
    - d2010: Dataframe with values from 2010
    - d2016: Dataframe with values from 2016
    """

    # Set the intervals to work with
    intervals = ['1997-2010', '1997-2016', '2010-2016']

    # Concatenate the data into one dataframe
    new_df = pd.concat([d1997, d2010, d2016], axis=1, keys=['1997', '2010', '2016'])

    # Add columns for the rates
    for interval in intervals:

        # Add a column with the change rate
        new_df[interval] = (new_df[interval[-4:]] - new_df[interval[0:4]])

        # Add a column with the smoothed trend
        new_col_name = interval + ' Trend'
        new_df[new_col_name] = sig.savgol_filter(x=new_df[interval], window_length=1001, polyorder=3)
        new_df[interval + ' FFT'] = fft_smooth(new_df[interval], 2002)

    return new_df


def dhigh_change_df(heights):
    """
    Calculate the average rate of dhigh change for each interval and its associated
    uncertainty. The uncertainty method is based on that used in Hapke et al (2012)
    but modified for the data outputted from Automorph
    """

    # Set the intervals to work with
    intervals = ['1997-2010', '1997-2016', '2010-2016']

    new_df = heights[['1997', '2010', '2016', '1997-2010', '1997-2016', '2010-2016']]

    # At this point, there is a column for the y-position of dhigh for every year.
    # Add columns for the rates.
    for interval in intervals:

        # Create new columns
        orig_df, begin_df, end_df, begin_time, end_time, dt = setup_change_df(interval, new_df)

        # Calculate the change rate between the years being analyzed
        change = end_df - begin_df
        new_df[interval + ' Change'] = change
        new_df[interval + ' Mean'] = np.nanmean(change)

        # Calculate uncertainty in the dune crest position
        up1, up2 = dhigh_uncertainty(begin_time, end_time, begin_df, end_df)

        # Calculate the rate uncertainty and regional uncertainty
        # based on the positional uncertainty
        position_uncertainty = np.sqrt((up1 ** 2) + (up2 ** 2))
        ur = rate_uncertainty(begin_time, end_time, position_uncertainty)
        region_uncertainty = regional_uncertainty(change, ur)
        new_df[interval + ' Mean Residual Error'] = mean_residual_error(new_df[interval + ' Mean Residual Error'], change)
        new_df[interval + ' Uncertainty'] = abs(ur)
        new_df[interval + ' Regional Uncertainty'] = abs(region_uncertainty)

    # Save the dataframe as a .csv
    save_name = os.path.join('..', 'Data', 'Dune Heights.csv')
    new_df.to_csv(save_name)

    # Return the new dataframe
    return new_df


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


def load_fence_locations(y):
    """
    Create a dataframe with profile numbers, category,
    and an associated y-value. The y-value is passed to the
    alongshore plotting function and is the y-axis position

    0 - Fort Macon
    1 - Non-fenced
    2 - Fenced
    """

    # Load the data. Can be from any year 2010-2016, results
    # won't change for this application
    data = load_data(2010, None)

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

    # Return the dataframe
    return data


def setup_change_df(interval, orig_df):
    """
    Prepare the dataframe for the change dataframes
    with uncertainty values

    Called by:
    - shoreline_change_df()
    """

    # Get the two times associated with the interval
    begin_time = str(interval[0:4])  # Initial time as a string
    end_time = str(interval[-4:])  # Final time as a string
    dt = (int(end_time) - int(begin_time))  # Time span as an int
    begin_df = orig_df[begin_time]  # Create a dataframe of shoreline positions from the initial time
    end_df = orig_df[end_time]  # Create a dataframe of shoreline positions from the final time

    # Create new column names
    new_col_name_1 = interval + ' Change'
    new_col_name_2 = interval + ' Mean'
    new_col_name_3 = interval + ' Uncertainty'
    new_col_name_4 = interval + ' Regional Uncertainty'
    new_col_name_5 = interval + ' Mean Residual Error'

    # Need to copy other columns before working
    orig_df[new_col_name_1] = orig_df['1997']
    orig_df[new_col_name_2] = orig_df['1997']
    orig_df[new_col_name_3] = orig_df['1997']
    orig_df[new_col_name_4] = orig_df['1997']
    orig_df[new_col_name_5] = orig_df['1997']

    return orig_df, begin_df, end_df, begin_time, end_time, dt


"""
Make the figure
"""


def bar_plots(data, fences, ax3, ax4, ax5, ax6, rate, text_y, bar_y, color, param):
    """
    This function makes a series of bar plots showing the change rates broken down by
    Fort Macon, non-fenced, and fenced locations as well as the island overall. Each location will
    have three bars for 1997-2016, 1997-2010, 2010-2016

    Called by;
    - make_overall_change_plot_w_kde()
    """

    conf_type = 95

    # Set colors and locations
    colors = ['red', 'blue', 'green']
    locations = ['Fort Macon', 'Non-fenced', 'Fenced']

    # Make a dataframe for all locations
    all_locs = pd.DataFrame({'1997-2010': data['1997-2010'].dropna(),
                             '2010-2016': data['2010-2016'].dropna(),
                             '1997-2016': data['1997-2016'].dropna()})
    all_locs['ID'] = 'All'
    all_locs = pd.melt(all_locs, id_vars=['ID'], value_vars=['1997-2010', '2010-2016', '1997-2016'])

    # Make a barplot of the data for all locations
    sns.barplot(x='ID', y='value', ci=conf_type, hue='variable', data=all_locs,
                palette=colors, estimator=np.nanmedian, errcolor='black', errwidth=2, ax=ax3)

    # Loop through and plot Fort Macon, non-fenced areas, and fenced areas
    for ii, ax in enumerate([ax4, ax5, ax6]):

        # Set temporary data
        use_data = data.loc[fences['Number'] == ii]

        # Make a dataframe for the location
        temp_locs = pd.DataFrame({'1997-2010': use_data['1997-2010'].dropna(),
                                  '2010-2016': use_data['2010-2016'].dropna(),
                                  '1997-2016': use_data['1997-2016'].dropna()})
        temp_locs['ID'] = locations[ii]
        temp_locs = pd.melt(temp_locs, id_vars=['ID'], value_vars=['1997-2010', '2010-2016', '1997-2016'])

        # Make a barplot of the data for the location
        sns.barplot(x='ID', y='value', ci=conf_type, hue='variable', data=temp_locs,
                    palette=colors, estimator=np.nanmedian, errcolor='black', errwidth=2, ax=ax)

    # Set common axis properties
    for ax in [ax3, ax4, ax5, ax6]:

        # Set grids
        utils.set_grid(ax)

        # Add a zero line
        ax.axhline(y=0, color='black', linewidth=2, linestyle='--', zorder=10)

        # Set the y-limits
        ax.set_ylim(bottom=-bar_y, top=bar_y)
        ax.set_yticks([-bar_y, (-bar_y)/2, 0, bar_y/2, bar_y])

        # ax.set_ylim(bottom=-25, top=bar_y)
        # ax.set_yticks([(-bar_y) / 2, 0, bar_y / 2, bar_y])

        # Set the y-axis
        if ax == ax3:
            if rate:
                ax.set_ylabel('Median ' + param + '\nChange Rate (m/yr)',
                              color=color,
                              fontname=opts['Font'],
                              fontsize=opts['Axis Font Size'],
                              fontweight=opts['Weight'])
            else:
                ax.set_ylabel('Median ' + param + '\nChange (m)',
                              color=color,
                              fontname=opts['Font'],
                              fontsize=opts['Axis Font Size'],
                              fontweight=opts['Weight'])
        else:
            ax.set_ylabel('')
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(left=False)

        # Remove the x-axis label
        ax.set_xlabel('')

        # Remove the legend
        ax.legend_.remove()

        # Set the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['bottom'].set_color(color)
        ax.tick_params(width=2)
        if ax == ax3:
            ax.spines['left'].set_linewidth(2)
            ax.spines['left'].set_color(color)
        else:
            ax.spines['left'].set_visible(False)

    # Add a "C" to ax3
    ax3.text(x=-0.35,
             y=bar_y - (bar_y * 0.20),
             s='C.',
             color='black',
             fontname=opts['Font'],
             fontsize=opts['Label Font Size'],
             fontweight=opts['Weight'])


def make_overall_change_plot_w_kde(data, fences, param, y_min, y_max, y_ticks, y_tick_labels,
                                   y_text, text_y, bar_y, spacing, rate=False, color='black'):
    """
    Make a plot of change rates for the pre-fencing (1997-2010),
    post-fencing (2010-2016), and overall (1997-2016) time periods

    Add a KDE plot in the second subplot

    Called by:
    - Shoreline_Change.py
    """

    # Setup the figure
    fig = plt.figure(dpi=opts['Figure DPI'])
    categories = ['Fort Macon', 'Non-Fenced', 'Fenced']
    gs = gridspec.GridSpec(nrows=2, ncols=4, wspace=0.0, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3], sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    ax5 = fig.add_subplot(gs[1, 2], sharey=ax3)
    ax6 = fig.add_subplot(gs[1, 3], sharey=ax3)

    # Setup additional plot styling
    lwidth = 2
    max_x = 0
    marker_shape = 'X'
    marker_alpha = 0.075
    x_max = 40000
    exes = [5000, 19300, 32000]
    yrs = ['1997-2016', '1997-2010', '2010-2016']
    labs = ['\n(Overall)', '\n(Pre-Fencing)', '\n(Post-Fencing)']
    text_years = ['1997-2010', '2010-2016', '1997-2016']
    text_labs = ['\n(Pre-Fencing)', '\n(Post-Fencing)', '\n(Overall)']

    # Add a grid for the top plots
    utils.set_grid([ax1, ax2])

    # Plot the data
    for ii, years in enumerate(['1997-2010', '1997-2016', '2010-2016']):

        # Scatter plot of the raw data
        sns.scatterplot(x=data.index,
                        y=years,
                        data=data,
                        linewidth=0,
                        color=opts['Marker Colors'][ii],
                        marker=marker_shape,
                        alpha=marker_alpha,
                        ax=ax1,
                        zorder=2)

        # Trend line over the raw data
        sns.lineplot(x=data.index,
                     y=years + ' Trend',
                     data=data,
                     linewidth=2,
                     color=opts['Trend Colors'][ii],
                     ax=ax1,
                     zorder=5)

        # KDE Plot in the second axis
        kde1 = sns.kdeplot(data[years].dropna(),
                           shade=True,
                           color=opts['Trend Colors'][ii],
                           linewidth=lwidth,
                           legend=False,
                           vertical=True,
                           ax=ax2,
                           zorder=2)

        # Add text labels
        ax1.text(x=exes[ii],
                 y=y_text - (2 * spacing),
                 s=text_years[ii] + text_labs[ii],
                 color=opts['Text Colors'][ii],
                 fontname=opts['Font'],
                 fontsize=opts['Label Font Size'] - 2,
                 fontweight=opts['Weight'],
                 ha='center',
                 va='center',
                 zorder=8)

        # Get the largest value in the KDE plot to set the axis limit later
        x1, y1 = kde1.get_lines()[0].get_data()
        if np.nanmax(x1) > max_x:
            max_x = np.around(a=np.nanmax(x1), decimals=2)

    # Add a zero-line
    ax1.axhline(y=0,
                xmin=0, xmax=x_max,
                color='black',
                linewidth=2,
                linestyle='--',
                zorder=7)
    ax2.axhline(y=0,
                xmin=0,
                xmax=max_x + 1,
                color='black',
                linewidth=2,
                linestyle='--',
                zorder=7)

    # Add an "A" and "B" label
    ax1.text(x=-0.5, y=y_max - (y_max * 0.20),
             s='A.',
             color='black',
             fontname=opts['Font'],
             fontsize=opts['Label Font Size'],
             fontweight=opts['Weight'])
    ax2.text(x=max_x - (max_x * 0.91), y=y_max - (y_max * 0.20),
             s='B.',
             color='black',
             fontname=opts['Font'],
             fontsize=opts['Label Font Size'],
             fontweight=opts['Weight'])

    # Set the x-axis
    ax1.set_xlim(right=x_max)
    ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
    ax1.set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '35', ''])
    ax1.set_xlabel('Alongshore Distance (km)',
                   color=color,
                   fontname=opts['Font'],
                   fontsize=opts['Axis Font Size'],
                   fontweight=opts['Weight'])

    # Set the y-axis
    ax1.set_ylim(bottom=y_min, top=y_max)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)
    if rate:
        ax1.set_ylabel(param + ' Change\nRate (m/yr)',
                       color=color,
                       fontname=opts['Font'],
                       fontsize=opts['Axis Font Size'],
                       fontweight=opts['Weight'])
    else:
        ax1.set_ylabel(param + ' Change (m)',
                       color=color,
                       fontname=opts['Font'],
                       fontsize=opts['Axis Font Size'],
                       fontweight=opts['Weight'])
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Add a barplot of change rates on the bottom of the figure
    bar_plots(data, fences, ax3, ax4, ax5, ax6, rate, text_y, bar_y, color, param)

    # Set the spines for the top plots
    utils.set_spines(axes=[ax1, ax2])

    # Set a tight layout and a transparent background
    utils.tight_and_transparent(axes=[ax1, ax2, ax3, ax4, ax5, ax6], fig=fig)

    # Save the figure
    title = 'Figure 7'
    utils.save_and_close(title)


"""
Functions to make a .txt file
"""


def make_stats(data, avg, fences, param):
    """
    Calculate change statistics about Bogue Banks

    - Mean, median rates for 1997-2010, 2010-2016, and 1997-2016
    - T-Test for 1997-2010 v. 2010-2016 v. 1997-2016
    """

    # Set the intervals to work with
    intervals = ['1997-2010', '1997-2016', '2010-2016']

    # Create a .txt file
    partial_fname = f'(Figure 7) {param} Change Statistics.txt'
    fname = os.path.join('..', 'Text Files', partial_fname)
    f = open(fname, 'w')

    # Write a header
    f.write('%s Change Statistics for Bogue Banks between 1997-2010, 2010-2016, 1997-2016\r\n' % param)
    f.write('\t- Summary statistics between Surveys\r\n')
    f.write('\t- KS-tests to compare pre- and post- fencing rates\r\n')
    f.write('\t- Mann-Whitney U-Tests between pre-fence, post-fence, and overall periods\r\n\r\n\r\n')

    # Write statistics for all of the profiles
    make_stats_section(f, data, avg, 'All Profiles', intervals)

    # Write stats for the different sections
    for ii, sect in enumerate(['Fort Macon', 'Non-Fenced', 'Fenced']):
        make_stats_section(f, data.loc[fences['Number'] == ii], avg.loc[fences['Number'] == ii], sect, intervals)

    # Compare Fort Macon to non-fenced to fenced
    mann_whitney_test_by_location(f, data, fences, bonferroni=3)

    # Close the file
    f.close()


def make_stats_section(f, data, avg, section, intervals):
    """
    Write a section in the stats file. This function is called by
    make_stats()

    Called by:
    - make_stats()
    """

    # Write a section header
    header_len = 35
    section_part = section + ' (n = ' + str(len(data)) + ')'
    lead_in = '#' * 2
    end_out = '#' * (header_len - len(section_part) - 4)
    f.write('%s\n' % ('#' * header_len))
    f.write('%s %s %s\n' % (lead_in, section_part, end_out))
    f.write('%s\n\n' % ('#' * header_len))

    # Write summary statistics for the data
    f.write('Summary Statistics (m/yr):\r\n')
    for yrs in intervals:
        f.write('%s' % (summary_statistics_alongshore(data, avg, yrs)))
    f.write('\r\n\r\n')

    # Perform an unequal variances T-test to compare the pre- and post-fencing rates to
    # each other. It is not important if the 1997-2016 is similar or dissimilar to either
    ks_test(f, data)

    f.write('\r\n\r\n')


def mann_whitney_test_by_location(f, data, fences, bonferroni=None):
    """
    Perform an unequal variances T-test and print the
    results to the stats file. This version compares
    Fort Macon v Non-Fenced v Fenced

    f: open text file with stats
    data: data to work on
    fences: dataframe with fence locations

    Called by:
    - make_stats()
    """

    bonferroni_alpha = 0.05 / 3

    # Set a list of intervals
    intervals = ['1997-2010', '1997-2016', '2010-2016']

    f.write('\r\n%s\r\n' % ('#' * 50))
    f.write('%s %s %s\r\n' % ('#' * 13, 'Compare locations', '#' * 18))
    f.write('%s\r\n\r\n' % ('#' * 50))

    # Write H-Test header
    f.write('Mann-Whitney U-Test with Bonferroni Correction (alpha = 0.05, bonf alpha = %f)\r\n' % bonferroni_alpha)
    f.write('H0: The sample populations have the same distribution\r\n')
    f.write('H1: The samples populations do not have the same distribution\r\n\r\n')

    for yrs in intervals:

        # Write a header for the years
        header_len = 25
        section_part = yrs
        lead_in = '#' * 2
        end_out = '#' * (header_len - len(section_part) - 4)
        f.write('%s\n' % ('#' * header_len))
        f.write('%s %s %s\n' % (lead_in, section_part, end_out))
        f.write('%s\n\n' % ('#' * header_len))

        # Perform tests
        u1, p1 = stats.mannwhitneyu(x=data[yrs].loc[fences['Number'] == 0], y=data[yrs].loc[fences['Number'] == 1],
                                    alternative='two-sided')
        u2, p2 = stats.mannwhitneyu(x=data[yrs].loc[fences['Number'] == 0], y=data[yrs].loc[fences['Number'] == 2],
                                    alternative='two-sided')
        u3, p3 = stats.mannwhitneyu(x=data[yrs].loc[fences['Number'] == 1], y=data[yrs].loc[fences['Number'] == 2],
                                    alternative='two-sided')

        # Print results
        if p1 < bonferroni_alpha:
            f.write('Fort Macon v. Non-Fenced:\tReject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
        else:
            f.write('Fort Macon v. Non-Fenced:\tFail to reject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n' % (u1, p1))
        if p2 < bonferroni_alpha:
            f.write('Fort Macon v. Fenced\t:\tReject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
        else:
            f.write('Fort Macon v. Fenced\t:\tFail to reject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n' % (u2, p2))
        if p3 < bonferroni_alpha:
            f.write('Non-Fenced v. Fenced\t:\tReject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n\r\n' % (u3, p3))
        else:
            f.write('Non-Fenced v. Fenced\t:\tFail to reject the null hypothesis\t(U = %0.4f\tp = %0.4f)\r\n\r\n' % (u3, p3))


def ks_test(f, data):
    """
    Perform a Kolmogorov-Smirnov test and print
    the results to the stats file. This version only compares
    years within one area (Fort Macon, non-fenced, fenced)

    f: open text file with stats
    data: data to compare

    Called by:
    - make_stats_section()
    """

    # Write KS-Test header
    f.write('Kolmogorov-Smirnov Test (alpha = 0.05)\r\n')
    f.write('H0: The sample populations come from the same distribution\r\n')
    f.write('H1: The samples populations do not come from the same distribution\r\n\r\n')

    d_stat, p_stat = stats.mannwhitneyu(data['1997-2010'], data['2010-2016'])
    d_round, p_round = str(np.around(d_stat, decimals=4)), str(np.around(p_stat, decimals=4))
    if p_stat <= 0.05:
        f.write('Reject the null hypothesis (D = %s\tp = %s)\r\n' % (d_round, p_round))
    else:
        f.write('Fail to reject the null hypothesis (D = %s\tp = %s)\r\n' % (d_round, p_round))


def summary_statistics_alongshore(data, avg, yrs):
    """
    Calculate summary statistics and format in a string

    - data: dataframe being worked on
    - avg: dataframe with uncertainties
    - yrs: years being looked at

    Called by:
    - make_stats_section()
    """

    # Set the number of decimals to round to
    dec = 1

    # Calculate the mean
    average = np.around(np.nanmean(avg[yrs + ' Change']), decimals=dec)

    # Calculate 2*sd
    sd_2 = np.around(np.std(avg[yrs + ' Change']), decimals=dec)

    # Calculate the median
    med = np.around(np.median(avg[yrs + ' Change']), decimals=dec)

    # Calculate the mode
    most = np.around(stats.mode(avg[yrs + ' Change'])[0][0], decimals=dec)

    # Calculate the mean uncertainty
    unc = np.around(np.nanmean(avg[yrs + ' Mean Residual Error']), decimals=2)
    min_unc = np.around(np.nanmin(avg[yrs + ' Mean Residual Error']), decimals=2)
    max_unc = np.around(np.nanmax(avg[yrs + ' Mean Residual Error']), decimals=2)
    avg_unc = np.around(np.nanmean(avg[yrs + ' Uncertainty']), decimals=2)

    # Calculate the percent increasing and decreasing
    pos = np.around((len(data[yrs][data[yrs] > 0]) / len(data[yrs])) * 100, decimals=dec)
    neg = np.around((len(data[yrs][data[yrs] < 0]) / len(data[yrs])) * 100, decimals=dec)

    # Calculate the 95% confidence interval
    x_bar = np.nanmean(data[yrs])  # Mean value
    s = np.nanstd(data[yrs])  # Standard deviation
    n = len(data[yrs])  # Sample size
    lo_conf = x_bar - (1.96 * (s / np.sqrt(n)))  # Lower bound of confidence interval
    hi_conf = x_bar + (1.96 * (s / np.sqrt(n)))  # Upper bound of confidence interval
    conf_int = np.around(hi_conf - lo_conf, decimals=dec)

    # Calculate the standard error on the mean
    # autocorr = estimated_autocorrelation(data[yrs])
    # sig_vals = np.where(autocorr <= 0.05)
    # n_star = np.nanmin(sig_vals)
    # sem = np.around(sd_2 / np.sqrt(n_star), decimals=dec)
    sem = np.around(stats.sem(data[yrs], nan_policy='omit'), decimals=dec)

    # Create a nicely formatted string
    res = f'{yrs}:\tMean: {average}\tSD: {sd_2}\tMedian: {med}\tMode: {most}\tMean Uncertainty: {avg_unc}\tMean MRE: {unc} ({min_unc} - {max_unc})\t95 Pct: {conf_int}\tPrograding Pct: {pos}\tEroding Pct: {neg}\tSEM: {sem}\r\n'

    return res


"""
Main function
"""


def main():
    """
    Main program function
    """

    # Load the data
    data_1997 = load_data(year=1997, variable='y_crest')
    data_2010 = load_data(year=2010, variable='y_crest')
    data_2016 = load_data(year=2016, variable='y_crest')

    # Make a dataframe of Dhighs
    heights = change_df(d1997=data_1997, d2010=data_2010, d2016=data_2016)
    heights_errors = dhigh_change_df(heights)

    # Load a dataframe with fence locations identified
    fences = load_fence_locations(y=0.5)

    # As a first-order guess at removing misidentified houses,
    # developed areas do not have natural dunes taller
    # than 8m, remove those for now. Ditto for dunes less than 3m
    heights['1997-2010'].loc[heights['1997-2010'] >= 3] = np.nan
    heights['1997-2010'].loc[(fences['Number'] > 0) & (heights['1997'] >= 8)] = np.nan
    heights['1997-2010'].loc[heights['1997'] <= 2.5] = np.nan
    heights['1997-2016'].loc[heights['1997-2016'] >= 3] = np.nan
    heights['1997-2016'].loc[(fences['Number'] > 0) & (heights['1997'] >= 8)] = np.nan
    heights['1997-2016'].loc[heights['1997'] <= 2.5] = np.nan

    # Make the figure
    y_min = -10
    y_text = y_min - (y_min * 0.35) + 3
    yticks = [y_min, (y_min) / 2, 0, (-y_min) / 2, -y_min]
    make_overall_change_plot_w_kde(data=heights,
                                   fences=fences,
                                   param='Dhigh',
                                   y_min=y_min,
                                   y_max=y_min * -1,
                                   y_ticks=yticks,
                                   y_tick_labels=[str(ii) for ii in yticks],
                                   text_y=40 * 0.80,
                                   bar_y=1.5,
                                   y_text=y_text,
                                   spacing=1.5,
                                   rate=False,
                                   color='black')

    # Calculate statistics
    make_stats(heights, heights_errors, fences, 'Dhigh')


if __name__ == '__main__':
    main()