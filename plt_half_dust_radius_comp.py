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
import matplotlib.tri as tri
from matplotlib import ticker, cm
import utilities as util
import sys
import h5py
import pandas as pd

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
masslim = 10**8

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

for reg, snap in reg_snaps:

    try:
        hdf = h5py.File("data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
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

        okinds1 = masses >= 10**9
        okinds2 = masses < 10 ** 9

        bins = np.logspace(0.08, 30, 50)
        print(bins)
        H, xbins, ybins = np.histogram2d(hdrs[okinds2], hlrs[okinds2],
                                         bins=bins, weights=w[okinds2])

        print(H)

        bin_wid = bins[1] - bins[0]
        xbin_cents = xbins[1:] - (bin_wid / 2)
        ybin_cents = ybins[1:] - (bin_wid / 2)

        XX, YY = np.meshgrid(xbin_cents, ybin_cents)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog()
        try:
            ax.hexbin(hdrs[okinds2], hlrs[okinds2], gridsize=50, mincnt=1,
                      C=w[okinds2], reduce_C_function=np.sum,
                      xscale='log', yscale='log',
                      norm=LogNorm(), linewidths=0.2, cmap='Greys')
            # cbar = ax.contourf(XX, YY, H, levels=10,
            #                    locator=ticker.LogLocator(),
            #                    norm=LogNorm(), cmap='Greys', alpha=0.8)
            cbar = ax.tricontour(hdrs[okinds2], hlrs[okinds2], w[okinds2], 5, linewidths=0.5, cmap='Greys', alpha=0.8, norm=LogNorm())
            # ax.hexbin(hdrs[okinds1], hlrs[okinds1], gridsize=50, mincnt=1, C=w[okinds1],
            #           reduce_C_function=np.sum, xscale='log', yscale='log',
            #           norm=LogNorm(), linewidths=0.2, cmap='viridis')
        except ValueError as e:
            print(e)
            continue

        min = np.min((ax.get_xlim(), ax.get_ylim()))
        max = np.max((ax.get_xlim(), ax.get_ylim()))

        ax.set_xlim([0.08, max])
        ax.set_ylim([0.08, max])

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

        fig.savefig('plots/' + str(z) + '/HalfDustRadius_' + f + '_'
                    + str(z) + '_' + Type + '_' + orientation + "_"
                    + extinction + "_" + '%d' % masslim
                    + "".replace(".", "p") + ".png",
                    bbox_inches='tight')

        plt.close(fig)

        ratio = hlrs / hdrs

        bins = np.logspace(0.08, 30, 50)
        ratio_bins = np.linspace(np.min(ratio[okinds2]),
                                 np.max(ratio[okinds2]), 50)

        H, xbins, ybins = np.histogram2d(hlrs[okinds2], ratio[okinds2],
                                         bins=(bins, ratio_bins),
                                         weights=w[okinds2])

        bin_wid = bins[1] - bins[0]
        xbin_cents = xbins[1:] - (bin_wid / 2)
        ybin_cents = ybins[1:] - (bin_wid / 2)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.loglog()
        try:
            cbar = ax.contourf((xbin_cents, ybin_cents), Z=H, levels=5,
                               norm=LogNorm(), cmap='Greys')
            # ax.hexbin(hlrs[okinds1], ratio[okinds1], gridsize=50, mincnt=1,
            #           C=w[okinds1], reduce_C_function=np.sum,
            #           xscale='log', yscale='log', norm=LogNorm(),
            #           linewidths=0.2, cmap='viridis')
        except ValueError as e:
            print(e)
            continue

        min = np.min((ax.get_xlim(), ax.get_ylim()))
        max = np.max((ax.get_xlim(), ax.get_ylim()))

        ax.set_xlim([0.08, max])
        ax.set_ylim([0.08, max])

        ax.plot([min, max], [1, 1], color='k', linestyle="--")

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ [pkpc]$")
        ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

        fig.savefig('plots/' + str(z) + '/HalfDustRadius_ratio_' + f + '_'
                    + str(z) + '_' + Type + '_' + orientation + "_"
                    + extinction + "_" + '%d' % masslim
                    + "".replace(".", "p") + ".png",
                    bbox_inches='tight')

        plt.close(fig)
