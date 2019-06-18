#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:27:12 2019

Figure 2: Map of Bogue Banks and its position along the
coast of North Carolina, showing town locations and the
location of Fort Macon State Park

The paper figure has the two maps arranged via Adobe
Illustrator. This code produces the two separate maps
as image files.

@author: michaelitzkin
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import haversine
import os


"""
Utility functions
"""


def draw_lat_lon(m, llon, llat, ulon, ulat, mticks, pticks):
    """
    Add parallels and meridians to the map
    """

    # Draw meridians
    meridians = np.round(np.arange(llon, ulon, (ulon - llon) / mticks),
                         decimals=2,
                         out=None)
    merids = m.drawmeridians(meridians,
                             labels=[False, False, False, True],
                             color='black',
                             linewidth=0.25,
                             fontsize=12,
                             fontweight='normal')
    set_color(merids, 'black')

    parallels = np.round(np.arange(llat, ulat, (ulat - llat) / pticks),
                         decimals=2,
                         out=None)
    pars = m.drawparallels(parallels,
                           labels=[True, False, False, False],
                           color='black',
                           linewidth=0.25,
                           fontsize=12,
                           fontweight='normal')
    set_color(pars, 'black')


def save_and_close(title):
    """
    Save the figure in the Figures folder
    and close the plot
    """
    save_name = os.path.join('..', 'Figures', title + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi='figure')
    plt.close()
    print(f'Figure saved: {save_name}\n')


def set_basemap(llat, llon, ulat, ulon, imagery):
    """
    Set the basemap layer of the figure

    The input arguments correspond to the
    desired bounding box
    """

    m = Basemap(projection='mill',
                llcrnrlat=llat,
                llcrnrlon=llon,
                urcrnrlat=ulat,
                urcrnrlon=ulon,
                resolution='h',
                epsg=4268)
    m.arcgisimage(service=imagery,
                  xpixels=1500,
                  verbose='True')

    return m


def set_color(x, color):
    """
    https://github.com/matplotlib/basemap/issues/145
    """
    for m in x:
        for t in x[m][1]:
            t.set_color(color)


def tight_and_transparent(ax, fig):
    """
    Set a tight layout and a transparent
    figure background
    """
    ax.tick_params(colors='black')
    plt.tight_layout()
    fig.patch.set_facecolor('w')
    fig.patch.set_alpha(0.0)


"""
Plotting functions
"""


def bogue_banks_map():
    """
    Make a map of Bogue Banks with towns, and water bodies marked off
    """

    # Store locations in dicts
    towns = {
        'Emerald Isle': [-77.041632, 34.661953],
        'Indian Beach': [-76.903616, 34.685105],
        'Pine Knoll Shores': [-76.813494, 34.697251],
        'Atlantic Beach': [-76.739851, 34.699932],
        'Fort Macon': [-76.696936, 34.694710],
    }

    inlets = {
        'Beaufort Inlet': [-76.6703154, 34.691648],
        'Bogue Inlet': [-77.111147, 34.644904],
    }

    other = {
        'Atlantic Ocean': [-76.86, 34.58],
        'Bogue Sound': [-76.908160, 34.705],
    }

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    llat_bogue, llon_bogue = 34.547749, -77.171629
    ulat_bogue, ulon_bogue = 34.735866, -76.560515
    dot_size = 10
    dot_alpha = 0.8
    dot_color = 'indianred'
    font = {'color': 'white',
            'name': 'Arial',
            'weight': 'bold',
            'size': 8}

    # Setup the basemap using the North Carolina bounding box
    # and the ESRI Imagery World 2D layer
    m = set_basemap(llat=llat_bogue,
                    llon=llon_bogue,
                    ulat=ulat_bogue,
                    ulon=ulon_bogue,
                    imagery='ESRI_Imagery_World_2D')

    # Loop through the towns and plot
    text_offset = 0.01
    for key in towns:
        ax.scatter(towns[key][0], towns[key][1],
                   s=dot_size,
                   c=dot_color,
                   alpha=dot_alpha)
        ax.text(x=str(float(towns[key][0])),
                y=str(float(towns[key][1]) - text_offset),
                s=key,
                rotation=-45,
                fontdict=font)

    # Loop through the inlets and plot
    for key in inlets:
        ax.scatter(inlets[key][0], inlets[key][1],
                   s=dot_size,
                   c=dot_color,
                   alpha=dot_alpha)
        ax.text(x=str(float(inlets[key][0])),
                y=str(float(inlets[key][1]) - text_offset),
                s=key,
                rotation=-45,
                fontdict=font)

    # Loop through other things and plot. Right
    # now this is just Bogue Sound and the Atlantic Ocean
    for key in other:
        ax.text(x=str(float(other[key][0])),
                y=str(float(other[key][1])),
                s=key,
                rotation=0,
                fontdict=font)

    # Add a scale
    offset_multiplier = 6.5
    lon0, lat0 = llon_bogue + text_offset, llat_bogue + text_offset
    lon1, lat1 = llon_bogue + (
            offset_multiplier * text_offset), llat_bogue + text_offset
    dist = haversine.haversine((lat0, lon0), (lat1, lon1),
                               miles=False)
    dist_str = str(int(dist)) + 'km'
    ax.plot([llon_bogue + text_offset,
             llon_bogue + (offset_multiplier * text_offset)],
            [llat_bogue + text_offset, llat_bogue + text_offset],
            color='white',
            linewidth=2)
    ax.text(
        x=((2 * llon_bogue) + (
                    offset_multiplier * text_offset)) / 2,
        y=llat_bogue + (1.5 * text_offset),
        s=dist_str,
        color='white',
        fontname='Arial',
        fontweight='bold',
        fontsize=9)

    # Add a north arrow
    ax.arrow(x=llon_bogue + (1.5 * text_offset),
             y=llat_bogue + (1.5 * text_offset),
             dx=0,
             dy=3 * text_offset,
             head_width=0.005,
             color='white')

    # Add latitude and longitude lines and markers
    draw_lat_lon(m=m,
                 llon=llon_bogue,
                 llat=llat_bogue,
                 ulon=ulon_bogue,
                 ulat=ulat_bogue,
                 mticks=5,
                 pticks=3)

    # Set a tight layout and make the background transparent
    tight_and_transparent(ax=ax, fig=fig)

    # Save the figure
    title = 'Figure 2 (Bogue Banks Map)'
    save_and_close(title=title)


def north_carolina_map():
    """
    Plot a map of North Carolina
    with county borders shown
    """

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    llat_nc, llon_nc = 33.586530, -78.924414
    ulat_nc, ulon_nc = 36.553161, -74.584814

    # Setup the basemap using the North Carolina bounding box
    # and the shaded relief layer
    m = set_basemap(llat=llat_nc,
                    llon=llon_nc,
                    ulat=ulat_nc,
                    ulon=ulon_nc,
                    imagery='World_Shaded_Relief')

    # Add state borders
    m.drawstates(linewidth=0.75, linestyle='solid', color='black')

    # Add counties
    m.drawcounties(linewidth=0.50, linestyle='solid', color='black')

    # Add coastlines
    m.drawcoastlines(linewidth=0.75, linestyle='solid', color='black')

    # Add a north arrow
    text_offset = 1
    ax.arrow(x=ulon_nc - (0.35 * text_offset),
             y=ulat_nc - (0.55 * text_offset),
             dx=0,
             dy=0.25 * text_offset,
             width=0.025,
             head_width=0.1,
             color='black',
             zorder=5)

    # Label North Carolina and the Atlantic Ocean
    lons, lats = [-78.50, -75.70], [35.50, 34.80]
    for ii, location in enumerate(['North Carolina', 'Atlantic Ocean']):
        ax.text(x=lons[ii], y=lats[ii],
                s=location,
                color='black',
                fontname='Arial',
                fontsize=14,
                fontweight='bold',
                zorder=5)

    # Add latitude and longitude lines and markers
    draw_lat_lon(m=m,
                 llon=llon_nc,
                 llat=llat_nc,
                 ulon=ulon_nc,
                 ulat=ulat_nc,
                 mticks=5,
                 pticks=5)

    # Set a tight layout, make the axes white and
    # make the background transparent
    tight_and_transparent(ax=ax, fig=fig)

    # Save the figure
    title = 'Figure 2 (North Carolina Map)'
    save_and_close(title=title)


"""
Main function
"""


def main():
    """
    Main program function
    """
    
    # Plot a map of North Carolina
    north_carolina_map()

    # Plot a map of Bogue Banks
    bogue_banks_map()
    
    
if __name__ == '__main__':
    main()
