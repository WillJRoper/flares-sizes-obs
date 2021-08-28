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
from flare.photom import M_to_lum
import h5py
import sys
import pandas as pd

sns.set_context("paper")
sns.set_style('whitegrid')

# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

if sys.argv[3] == "All":
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
else:
    snaps = sys.argv[3]

# Define filter
filters = ['FAKE.TH.'+ f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H', 'K']]

csoft = 0.001802390 / (0.6777) * 1e3

masslim = 10 ** 8.0

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
nstar_dict = {}
weight_dict = {}

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

for reg, snap in reg_snaps:

    hdf = h5py.File("data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
                                                                orientation),
                    "r")

    lumin_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    nstar_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    for f in filters:

        lumin_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        nstar_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = hdf[f]["Mass"][...]
        okinds = masses > masslim

        print(reg, snap, f, masses[okinds].size)

        lumin_dict[snap][f].extend(
            hdf[f]["Luminosity"][...][okinds])
        mass_dict[snap][f].extend(masses[okinds])
        try:
            nstar_dict[snap][f].extend(hdf[f]["nStar"][...])
        except KeyError:
            continue
        weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))

    hdf.close()

# Set mass limit
masslim = 10 ** 9

for f in filters:

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)

    for snap in snaps:

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        mass = np.array(mass_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        nstar = np.array(nstar_dict[snap][f])
        w = np.array(weight_dict[snap][f])
        print(len(mass), len(lumins), len(w))

        okinds1 = mass < masslim
        okinds2 = mass >= masslim

        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            print(len(mass[okinds1]), len(lumins[okinds1]), len(w[okinds1]))
            print(len(mass[okinds2]), len(lumins[okinds2]), len(w[okinds2]))
            cbar = ax.hexbin(mass[okinds1], lumins[okinds1],
                             gridsize=50, mincnt=1, C=w[okinds1],
                             reduce_C_function=np.sum,
                             xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2,
                             cmap='Greys', alpha=0.6)
            cbar = ax.hexbin(mass[okinds2], lumins[okinds2],
                             gridsize=50, mincnt=1, C=w[okinds2],
                             reduce_C_function=np.sum,
                             xscale='log', yscale='log',
                             norm=LogNorm(), linewidths=0.2,
                             cmap='viridis', alpha=0.9)
        except ValueError as e:
            print(e)
            continue

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_ylabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
        ax.set_xlabel('$M_\star/ M_\odot$')

        ax.tick_params(axis='both', which='minor', bottom=True, left=True)

        fig.savefig(
            'plots/' + str(z) + '/MassLumin_' + f + '_' + str(
                z) + '_'
            + orientation + '_' + Type + "_" + extinction + "_"
            + '%d.png' % np.log10(masslim),
            bbox_inches='tight')

        plt.close(fig)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # try:
        #     cbar = ax.hexbin(mass, nstar,
        #                      gridsize=50, mincnt=1, C=w,
        #                      reduce_C_function=np.sum,
        #                      xscale='log', yscale='log',
        #                      norm=LogNorm(), linewidths=0.2,
        #                      cmap='viridis', alpha=0.9)
        #     ax.axhline(nlim, linestyle="--", color="k", alpha=0.7)
        # except ValueError as e:
        #     print(e)
        #     continue
        #
        # ax.text(0.95, 0.05, f'$z={z}$',
        #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
        #                   ec="k", lw=1, alpha=0.8),
        #         transform=ax.transAxes, horizontalalignment='right',
        #         fontsize=8)
        #
        # # Label axes
        # ax.set_ylabel(r'$N_\star$')
        # ax.set_xlabel('$M_\star/ M_\odot$')
        #
        # ax.tick_params(axis='both', which='minor', bottom=True, left=True)
        #
        # fig.savefig(
        #     'plots/' + str(z) + '/MassNStar_' + f + '_' + str(
        #         z) + '_'
        #     + orientation + '_' + Type + "_" + extinction + "_"
        #     + '%d.png' % nlim,
        #     bbox_inches='tight')
        #
        # plt.close(fig)
