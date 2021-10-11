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
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

sns.set_context("paper")
sns.set_style('whitegrid')


def mass_lumin(mass, lumins, nokinds, okinds1, okinds2, w,
               f, snap, orientation, Type, extinction):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Snapshot =", snap)
    print("Filter =", f)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    okinds1 = np.logical_and(nokinds, okinds1)
    okinds2 = np.logical_and(nokinds, okinds2)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=(3, 10), width_ratios=(10, 3))
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0])
    axright = fig.add_subplot(gs[1, 1])
    axtop.loglog()
    axright.loglog()
    axtop.grid(False)
    axright.grid(False)
    try:
        cbar = ax.hexbin(mass, lumins,
                         gridsize=50, mincnt=1, C=w,
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='Greys', alpha=0.2)
        cbar = ax.hexbin(mass[okinds1], lumins[okinds1],
                         gridsize=50, mincnt=1, C=w[okinds1],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='Greys', alpha=0.8)
        cbar = ax.hexbin(mass[okinds2], lumins[okinds2],
                         gridsize=50, mincnt=1, C=w[okinds2],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='viridis', alpha=0.9)
    except ValueError as e:
        print(e)

    lumin_bins = np.logspace(27.2, 31.5, 100)
    Hbot2_all, bin_edges = np.histogram(lumins, bins=lumin_bins)
    Hbot2, bin_edges = np.histogram(lumins[nokinds], bins=lumin_bins)
    lbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    print("Complete to", lbin_cents[np.argmin(Hbot2_all - Hbot2)])

    axright.plot(Hbot2_all, lbin_cents, color="k", alpha=0.4)
    axright.plot(Hbot2, lbin_cents, color="k")

    mass_bins = np.logspace(7.5, 11.5, 100)
    Htop2_all, bin_edges = np.histogram(mass, bins=mass_bins)
    Htop2, bin_edges = np.histogram(mass[nokinds], bins=mass_bins)
    mbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    print("Complete to", mbin_cents[np.argmin(Htop2_all - Htop2)])

    axtop.plot(mbin_cents, Htop2_all, color="k", alpha=0.4)
    axtop.plot(mbin_cents, Htop2, color="k")

    # Remove axis labels and ticks
    axtop.tick_params(axis='x', top=False, bottom=False,
                      labeltop=False, labelbottom=False)
    axtop.tick_params(axis='y', left=False, right=False,
                      labelleft=False, labelright=False)
    axright.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
    axright.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

    axtop.spines['top'].set_visible(False)
    axtop.spines['right'].set_visible(False)
    axtop.spines['left'].set_visible(False)

    axright.spines['bottom'].set_visible(False)
    axright.spines['top'].set_visible(False)
    axright.spines['right'].set_visible(False)

    # ax.text(0.95, 0.05, f'$z={z}$',
    #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
    #                   ec="k", lw=1, alpha=0.8),
    #         transform=ax.transAxes, horizontalalignment='right',
    #         fontsize=8)

    # Label axes
    ax.set_ylabel(r"$L_{" + f.split(".")[-1] + "}/$ [erg $/$ s $/$ Hz]")
    ax.set_xlabel('$M_\star/ M_\odot$')

    ax.tick_params(axis='both', which='minor', bottom=True, left=True)

    ax.set_xlim(10 ** 7.5, 10 ** 11.5)
    axtop.set_xlim(10 ** 7.5, 10 ** 11.5)
    ax.set_ylim(10 ** 27.2, 10 ** 31.5)
    axright.set_ylim(10 ** 27.2, 10 ** 31.5)

    fig.savefig(
        'plots/' + str(z) + '/MassLumin_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + '.png',
        bbox_inches='tight')

    plt.close(fig)


if __name__ == "__main__":

    # Set orientation
    orientation = "sim"

    snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
             '010_z005p000']
    all_snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
                 '006_z009p000', '007_z008p000', '008_z007p000',
                 '009_z006p000', '010_z005p000', '011_z004p770']

    # Define filter
    # filters = ['FAKE.TH.' + f
    #            for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
    #                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
    filters = ['FAKE.TH.' + f for f in ['FUV', 'MUV', 'NUV']]

    keys = ["Mass", "Image_Luminosity", "Luminosity","nStar"]

    csoft = 0.001802390 / (0.6777) * 1e3

    data = {}
    intr_data = {}

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

        try:
            hdf = h5py.File(
                "data/flares_sizes_all_{}_{}_{}_{}.hdf5".format(reg, snap, "Total",
                                                            orientation),
                "r")
        except OSError as e:
            print(e)
            continue

        data.setdefault(snap, {})
        intr_data.setdefault(snap, {})

        for f in filters:

            print(reg, snap, f)

            data[snap].setdefault(f, {})
            intr_data[snap].setdefault(f, {})

            for key in keys:
                data[snap][f].setdefault(key, []).extend(hdf[f][key][...])

            data[snap][f].setdefault("Weight", []).extend(
                np.full(hdf[f]["Mass"][...].size, weights[int(reg)]))

        hdf.close()

        try:
            hdf = h5py.File(
                "data/flares_sizes_all_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                            "Intrinsic",
                                                            orientation),
                "r")
        except OSError as e:
            print(e)
            continue

        for f in filters:

            surf_dens = hdf[f]["Image_Luminosity"][...] \
                        / (np.pi * (2 * hdf[f]["HLR_0.5"][...]) ** 2)

            intr_data[snap][f].setdefault("Inner_Surface_Density",
                                          []).extend(surf_dens)

            for key in keys:
                intr_data[snap][f].setdefault(key, []).extend(hdf[f][key][...])

        hdf.close()

    for snap in snaps:

        for f in filters:

            print(snap, f)

            for key in data[snap][f].keys():
                data[snap][f][key] = np.array(data[snap][f][key])

            for key in intr_data[snap][f].keys():
                intr_data[snap][f][key] = np.array(intr_data[snap][f][key])

            okinds = np.logical_and(
                intr_data[snap][f]["Inner_Surface_Density"] > 10 ** 26,
                intr_data[snap][f]["nStar"] > 100)

            data[snap][f]["okinds"] = okinds
            intr_data[snap][f]["okinds"] = okinds

            compact_pop = np.array(
                intr_data[snap][f]["Inner_Surface_Density"]) >= 10 ** 29
            diffuse_pop = np.array(
                intr_data[snap][f]["Inner_Surface_Density"]) < 10 ** 29

            data[snap][f]["Compact_Population"] = compact_pop
            data[snap][f]["Diffuse_Population"] = diffuse_pop
            intr_data[snap][f]["Compact_Population"] = compact_pop
            intr_data[snap][f]["Diffuse_Population"] = diffuse_pop

    for f in filters:
        for snap in snaps:
            print("---------------------------", f, snap,
                  "---------------------------")
            mass_lumin(intr_data[snap][f]["Mass"],
                       intr_data[snap][f]["Luminosity"],
                       intr_data[snap][f]["okinds"],
                       intr_data[snap][f]["Diffuse_Population"],
                       intr_data[snap][f]["Compact_Population"],
                       data[snap][f]["Weight"],
                       f, snap, orientation, "Intrinsic", "default")
