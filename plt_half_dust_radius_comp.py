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

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Set orientation
orientation = sys.argv[1]
Type = sys.argv[2]

# Define luminosity and mathrm{dust} model types
extinction = 'default'

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV', 'FAKE.TH.V')

# Define dictionaries for results
hlr_dict = {}
hdr_dict = {}
mass_dict = {}
weight_dict = {}

# Set mass limit
masslim = 10 ** 8

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

for reg, snap in reg_snaps:

    try:
        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
                                                        orientation),
            "r")
    except OSError as e:
        print(e)
        continue

    hdr_dict.setdefault(snap, {})
    hlr_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hdr_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = hdf[f]["Mass"][...]
        mass_dict[snap][f].extend(masses)

        hlr_dict[snap][f].extend(hdf[f]["HLR_0.5"][...])
        hdr_dict[snap][f].extend(hdf[f]["HDR"][...])
        weight_dict[snap][f].extend(np.full(masses.size, weights[int(reg)]))

    hdf.close()

for snap in snaps:

    for f in filters:

        print("Plotting for:")
        print("Orientation =", orientation)
        print("Filter =", f)
        print("Snapshot =", snap)

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hdrs = np.array(hdr_dict[snap][f])
        hlrs = np.array(hlr_dict[snap][f])
        masses = np.array(mass_dict[snap][f])
        w = np.array(weight_dict[snap][f])

        okinds = np.logical_and(np.logical_and(hdrs > 0, hlrs > 0), masses > 0)

        hdrs = hdrs[okinds]
        hlrs = hlrs[okinds]
        masses = masses[okinds]
        w = w[okinds]

        okinds1 = masses >= 10 ** 9
        okinds2 = masses < 10 ** 9

        bins = np.logspace(np.log10(0.08), np.log10(20), 40)
        H, xbins, ybins = np.histogram2d(hdrs[okinds2], hlrs[okinds2],
                                         bins=bins, weights=w[okinds2])

        # Resample your data grid by a factor of 3 using cubic spline interpolation.
        H = scipy.ndimage.zoom(H, 3)

        # percentiles = [np.min(w),
        #                10**-3,
        #                10**-1,
        #                1, 2, 5]

        percentiles = [np.percentile(H, 80),
                       np.percentile(H, 90),
                       np.percentile(H, 95),
                       np.percentile(H, 99)]

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
                             mincnt=1,
                             C=w[okinds2], reduce_C_function=np.sum,
                             xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='Greys',
                             alpha=0.7)
            ax.hexbin(hdrs[okinds1], hlrs[okinds1], gridsize=50, mincnt=1,
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
            continue

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
                    + extinction + "_" + '%d' % masslim
                    + "".replace(".", "p") + ".png",
                    bbox_inches='tight')

        plt.close(fig)

        ratio = hlrs / hdrs

        bins = np.logspace(np.log10(0.08), np.log10(40), 40)

        H, xbins, ybins = np.histogram2d(hlrs[okinds2], ratio[okinds2],
                                         bins=bins,
                                         weights=w[okinds2])

        # Resample your data grid by a factor of 3 using cubic spline interpolation.
        H = scipy.ndimage.zoom(H, 3)

        # percentiles = [np.min(w),
        #                10**-3,
        #                10**-1,
        #                1, 2, 5]

        percentiles = [np.percentile(H, 80),
                       np.percentile(H, 90),
                       np.percentile(H, 95),
                       np.percentile(H, 99)]

        bins = np.logspace(np.log10(0.08), np.log10(40), H.shape[0] + 1)

        xbin_cents = (bins[1:] + bins[:-1]) / 2
        ybin_cents = (bins[1:] + bins[:-1]) / 2

        XX, YY = np.meshgrid(xbin_cents, ybin_cents)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog()
        try:
            cbar = ax.hexbin(hlrs[okinds2], ratio[okinds2], gridsize=50,
                             mincnt=1,
                             C=w[okinds2], reduce_C_function=np.sum,
                             xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2, cmap='Greys',
                             alpha=0.7)
            ax.hexbin(hlrs[okinds1], ratio[okinds1], gridsize=50, mincnt=1,
                      C=w[okinds1], reduce_C_function=np.sum,
                      xscale='log', yscale='log', norm=LogNorm(),
                      linewidths=0.2, cmap='viridis', alpha=0.8)
            cbar = ax.contour(XX, YY, H.T, levels=percentiles,
                              norm=LogNorm(), cmap=cmr.bubblegum_r,
                              linewidth=2)
        except ValueError as e:
            print(e)
            continue

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
                    + extinction + "_" + '%d' % masslim
                    + "".replace(".", "p") + ".png",
                    bbox_inches='tight')

        plt.close(fig)
