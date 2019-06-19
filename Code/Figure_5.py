#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 19, 2019

Figure 5: Map of Bogue Banks showing Lidar profile locations color
coded according to the presence or absence of sand fences

@author: michaelitzkin
"""

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import haversine
import os


def load_fences():
    """
    Load the sand fence locations and set the
    color to plot as based on if the profiles are
    fenced, non-fenced, or in Fort Macon
    """

    # Set the colors
    colors = ['darkred', 'dimgray', 'cornflowerblue']

    # Load the data
    path = os.path.join('..', 'Data', 'Morphometrics for Bogue 2014.csv')
    data = pd.read_csv(path)

    # Add a column to the data table with an index based on the fencing
    # scenario
    fort_macon = -2229
    data['Area'] = 0                                               # Fort Macon
    data['Area'][:fort_macon].loc[np.isnan(data['x_fence'])] = 1   # Non-fenced
    data['Area'][:fort_macon].loc[~np.isnan(data['x_fence'])] = 2  # Fenced

    data['Area'].loc[data['Area'] == 0] = colors[0]
    data['Area'].loc[data['Area'] == 1] = colors[1]
    data['Area'].loc[data['Area'] == 2] = colors[2]

    return data


def setcolor(x, color):
    """
    https://github.com/matplotlib/basemap/issues/145
    """
    for m in x:
        for t in x[m][1]:
            t.set_color(color)


def bogue_banks_fence_map(df):
    """
    Plot a satellite view of Bogue Banks with all the profile
    locations plotted and colored based on the presence of
    sand fences

    df: DataFrame with data to plot
    """

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    colors = ['darkred', 'dimgray', 'cornflowerblue']

    # Set the bounding box for Bogue Banks
    llat_bogue, llon_bogue = 34.547749, -77.171629
    ulat_bogue, ulon_bogue = 34.735866, -76.560515

    # Setup the basemap using the North Carolina bounding box
    # and the ESRI Imagery World 2D layer
    m = Basemap(projection='mill',
                llcrnrlat=llat_bogue, llcrnrlon=llon_bogue,
                urcrnrlat=ulat_bogue, urcrnrlon=ulon_bogue,
                resolution='h', epsg=4268)
    m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=1500,
                  verbose='True')

    # Plot the data
    ax.scatter(x=df['mhw_lon'],
               y=df['mhw_lat'] - 0.01,
               c=df['Area'],
               s=10)

    # Add text labels
    x_text, y_text, sep = -76.70, 34.60, 0.02
    levels = ['Fort Macon', 'Non-Fenced', 'Fenced']
    for i in range(0, 3):
        ax.text(x=x_text,
                y=y_text - (i * sep),
                s=levels[i],
                color=colors[i],
                fontname='Arial',
                fontsize=14,
                fontweight='bold',
                zorder=5)

    # Add a box behind the text
    rect = Rectangle(xy=(x_text - 0.02, y_text - (sep * 3)),
                     width=ulon_bogue - x_text + 0.05,
                     height=(sep * 2) + 0.050,
                     fill=True,
                     facecolor='white',
                     edgecolor='black',
                     zorder=3)
    ax.add_patch(rect)

    # Add a scale
    offset_multiplier = 6.5
    lon0, lat0 = llon_bogue + 0.01, llat_bogue + 0.01
    lon1, lat1 = llon_bogue + (offset_multiplier * 0.01), llat_bogue + 0.01
    dist = haversine.haversine((lat0, lon0), (lat1, lon1), miles=False)
    dist_str = str(int(dist)) + 'km'
    ax.plot([llon_bogue + 0.01,
             llon_bogue + (offset_multiplier * 0.01)],
            [llat_bogue + 0.01, llat_bogue + 0.01],
            color='white',
            linewidth=2)
    ax.text(
        x=((2 * llon_bogue) + (offset_multiplier * 0.01)) / 2,
        y=llat_bogue + (1.5 * 0.01),
        s=dist_str,
        color='white',
        fontname='Arial',
        fontweight='bold',
        fontsize=9)

    # Add a north arrow
    ax.arrow(x=llon_bogue + (1.5 * 0.01),
             y=llat_bogue + (1.5 * 0.01),
             dx=0,
             dy=3 * 0.01,
             head_width=0.005,
             color='white')

    # Add latitude and longitude lines and markers
    ticks_merid = 5
    ticks_pars = 4
    meridians = np.round(np.arange(llon_bogue, ulon_bogue, (ulon_bogue - llon_bogue) / ticks_merid), decimals=2, out=None)
    parallels = np.round(np.arange(llat_bogue, ulat_bogue, (ulat_bogue - llat_bogue) / ticks_pars), decimals=2, out=None)
    pars = m.drawparallels(parallels,
                           labels=[True, False, False, False],
                           color='white',
                           linewidth=0.01,
                           fontsize=12,
                           fontweight='normal')
    merids = m.drawmeridians(meridians,
                             labels=[False, False, False, True],
                             color='white',
                             linewidth=0.01,
                             fontsize=12,
                             fontweight='normal')
    setcolor(pars, 'black')
    setcolor(merids, 'black')

    # Set a tight layout, white ticks and
    # make the background transparent
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(0.0)

    # Save the figure
    title = 'Figure 5'
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


def main():
    """
    Main program function
    """

    # Load the data
    data = load_fences()

    # Make the figure
    bogue_banks_fence_map(df=data)


if __name__ == '__main__':
    main()