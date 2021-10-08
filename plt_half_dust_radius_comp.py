#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import scipy.ndimage
from matplotlib import ticker
import sys
import h5py
import pandas as pd
import cmasher as cmr

sns.set_context("paper")
sns.set_style('whitegrid')

def hdr_comp(hdrs, hlrs, hlrints, w, okinds, okinds1, okinds2, f, orientation, snap, Type, extinction):

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Filter =", f)
    print("Snapshot =", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    okinds1 = np.logical_and(okinds, okinds1)
    okinds2 = np.logical_and(okinds, okinds2)

    bins = np.logspace(np.log10(0.08), np.log10(20), 40)
    H, xbins, ybins = np.histogram2d(hdrs[okinds2], hlrs[okinds2],
                                     bins=bins, w=w[okinds2])

    # Resample your data grid by a factor of 3 using cubic spline interpolation.
    H = scipy.ndimage.zoom(H, 3)

    # percentiles = [np.min(w),
    #                10**-3,
    #                10**-1,
    #                1, 2, 5]

    try:
        percentiles = [np.percentile(H[H > 0], 50),
                       np.percentile(H[H > 0], 80),
                       np.percentile(H[H > 0], 90),
                       np.percentile(H[H > 0], 95),
                       np.percentile(H[H > 0], 99)]
    except IndexError:
        

    bins = np.logspace(np.log10(0.08), np.log10(20), H.shape[0] + 1)

    xbin_cents = (bins[1:] + bins[:-1]) / 2
    ybin_cents = (bins[1:] + bins[:-1]) / 2

    XX, YY = np.meshgrid(xbin_cents, ybin_cents)

    print(percentiles)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    try:
        cbar = ax.hexbin(hdrs[okinds2], hlrs[okinds2], gridsize=50,
                         mincnt=np.min(w),
                         C=w[okinds2], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='Greys',
                         alpha=0.7)
        ax.hexbin(hdrs[okinds1], hlrs[okinds1], gridsize=50, mincnt=np.min(w),
                  C=w[okinds1],
                  reduce_C_function=np.sum, xscale='log', yscale='log',
                  norm=LogNorm(), linewidths=0.2, cmap='viridis',
                  alpha=0.8)
        cbar = ax.contour(XX, YY, H.T, levels=percentiles,
                          locator=ticker.LogLocator(),
                          norm=LogNorm(), cmap=cmr.bubblegum_r,
                          linewidth=2)
    except ValueError as e:
        print(e)
        

    min = np.min((ax.get_xlim(), ax.get_ylim()))
    max = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.plot([min, max], [min, max], color='k', linestyle="--")

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ [pkpc]$")
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    plt.axis('scaled')

    ax.set_xlim([0.08, 20])
    ax.set_ylim([0.08, 20])

    # fig.colorbar(cbar)

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight')

    plt.close(fig)

    ratio = hlrs / hdrs

    bins = np.logspace(np.log10(0.08), np.log10(40), 40)

    H, xbins, ybins = np.histogram2d(hdrs[okinds2], ratio[okinds2],
                                     bins=bins,
                                     w=w[okinds2])

    # Resample your data grid by a factor of 3 using cubic spline interpolation.
    H = scipy.ndimage.zoom(H, 3)

    # percentiles = [np.min(w),
    #                10**-3,
    #                10**-1,
    #                1, 2, 5]

    try:
        percentiles = [np.percentile(H[H > 0], 50),
                       np.percentile(H[H > 0], 80),
                       np.percentile(H[H > 0], 90),
                       np.percentile(H[H > 0], 95),
                       np.percentile(H[H > 0], 99)]
    except IndexError:
        

    bins = np.logspace(np.log10(0.08), np.log10(40), H.shape[0] + 1)

    xbin_cents = (bins[1:] + bins[:-1]) / 2
    ybin_cents = (bins[1:] + bins[:-1]) / 2

    XX, YY = np.meshgrid(xbin_cents, ybin_cents)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    try:
        cbar = ax.hexbin(hdrs[okinds2], ratio[okinds2], gridsize=50,
                         mincnt=np.min(w),
                         C=w[okinds2], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='Greys',
                         alpha=0.7)
        ax.hexbin(hdrs[okinds1], ratio[okinds1], gridsize=50, mincnt=np.min(w),
                  C=w[okinds1], reduce_C_function=np.sum,
                  xscale='log', yscale='log', norm=LogNorm(),
                  linewidths=0.2, cmap='viridis', alpha=0.8)
        cbar = ax.contour(XX, YY, H.T, levels=percentiles,
                          norm=LogNorm(), cmap=cmr.bubblegum_r,
                          linewidth=2)
    except ValueError as e:
        print(e)
        

    min = np.min((ax.get_xlim(), ax.get_ylim()))
    max = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.plot([min, max], [1, 1], color='k', linestyle="--")

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ R_{1/2, dust}$")
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    ax.set_xlim([0.08, 20])
    ax.set_ylim([0.08, 50])

    plt.axis('scaled')

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_ratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight')

    plt.close(fig)

    # ==================================================================

    ratio = hlrs / hlrints

    bins = np.logspace(np.log10(0.08), np.log10(40), 40)

    H, xbins, ybins = np.histogram2d(hdrs[okinds2], ratio[okinds2],
                                     bins=bins,
                                     w=w[okinds2])

    # Resample your data grid by a factor of 3 using cubic spline interpolation.
    H = scipy.ndimage.zoom(H, 3)

    # percentiles = [np.min(w),
    #                10**-3,
    #                10**-1,
    #                1, 2, 5]

    try:
        percentiles = [np.percentile(H[H > 0], 50),
                       np.percentile(H[H > 0], 80),
                       np.percentile(H[H > 0], 90),
                       np.percentile(H[H > 0], 95),
                       np.percentile(H[H > 0], 99)]
    except IndexError:
        

    bins = np.logspace(np.log10(0.08), np.log10(40), H.shape[0] + 1)

    xbin_cents = (bins[1:] + bins[:-1]) / 2
    ybin_cents = (bins[1:] + bins[:-1]) / 2

    XX, YY = np.meshgrid(xbin_cents, ybin_cents)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    try:
        cbar = ax.hexbin(hdrs[okinds2], ratio[okinds2], gridsize=50,
                         mincnt=np.min(w),
                         C=w[okinds2], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2, cmap='Greys',
                         alpha=0.7)
        ax.hexbin(hdrs[okinds1], ratio[okinds1], gridsize=50, mincnt=np.min(w),
                  C=w[okinds1], reduce_C_function=np.sum,
                  xscale='log', yscale='log', norm=LogNorm(),
                  linewidths=0.2, cmap='viridis', alpha=0.8)
        cbar = ax.contour(XX, YY, H.T, levels=percentiles,
                          norm=LogNorm(), cmap=cmr.bubblegum_r,
                          linewidth=2)
    except ValueError as e:
        print(e)
        

    min = np.min((ax.get_xlim(), ax.get_ylim()))
    max = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.plot([min, max], [1, 1], color='k', linestyle="--")

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2,"
                  + f.split(".")[-1]
                  + ", \mathrm{Attenuated}}/ R_{1/2,"
                  + f.split(".")[-1] + ", \mathrm{Intrinsic}}$")
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    ax.set_xlim([0.08, 20])
    ax.set_ylim([0.08, 50])

    plt.axis('scaled')

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_hlrratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight')

    plt.close(fig)
