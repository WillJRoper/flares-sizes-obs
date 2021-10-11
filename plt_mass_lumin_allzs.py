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
               f, orientation, Type, extinction):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    okinds1 = np.logical_and(nokinds, okinds1)
    okinds2 = np.logical_and(nokinds, okinds2)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=(2, 10), width_ratios=(10, 2))
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0])
    axright = fig.add_subplot(gs[1, 1])
    axtop.loglog()
    axright.loglog()
    axtop.grid(False)
    axright.grid(False)
    try:
        cbar = ax.hexbin(mass[~nokinds],
                         lumins[~nokinds],
                         gridsize=50, mincnt=1,
                         C=w[~nokinds],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='Greys', alpha=0.2,
                         extent=(7.5, 11.5, 26.3, 31.5))
        cbar = ax.hexbin(mass[okinds1], lumins[okinds1],
                         gridsize=50, mincnt=1, C=w[okinds1],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='Greys',
                         extent=(7.5, 11.5, 26.3, 31.5))
        cbar = ax.hexbin(mass[okinds2], lumins[okinds2],
                         gridsize=50, mincnt=1, C=w[okinds2],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='viridis',
                         extent=(7.5, 11.5, 26.3, 31.5))
    except ValueError as e:
        print(e)

    lumin_bins = np.logspace(26.3, 31.5, 150)
    Hbot2_all, bin_edges = np.histogram(lumins, bins=lumin_bins)
    Hbot2, bin_edges = np.histogram(lumins[nokinds],
                                    bins=lumin_bins)
    lbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    comp_l = np.max(lumins[~nokinds])
    print("Complete to log_10(L/[erg s^-1 Hz^-1]) =", np.log10(comp_l))

    axright.plot(Hbot2_all, lbin_cents, color="k", alpha=0.4)
    axright.plot(Hbot2, lbin_cents, color="k")
    axright.axhline(comp_l, linestyle="--", alpha=0.6, color="k")
    ax.axhline(comp_l, linestyle="--", alpha=0.6, color="k")

    mass_bins = np.logspace(7.5, 11.5, 50)
    Htop2_all, bin_edges = np.histogram(mass, bins=mass_bins)
    Htop2, bin_edges = np.histogram(mass[nokinds], bins=mass_bins)
    mbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    comp_m = np.max(mass[~nokinds])
    print("Complete to log_10(M/M_sun) =", np.log10(comp_m))

    axtop.plot(mbin_cents, Htop2_all, color="k", alpha=0.4)
    axtop.plot(mbin_cents, Htop2, color="k")
    axtop.axvline(comp_m, linestyle="--", alpha=0.6, color="k")
    ax.axvline(comp_m, linestyle="--", alpha=0.6, color="k")

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
    ax.set_ylim(10 ** 26.3, 10 ** 31.5)
    axright.set_ylim(10 ** 26.3, 10 ** 31.5)

    fig.savefig(
        'plots/MassLumin_allz_' + f + '_'
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

    keys = ["Mass", "Image_Luminosity", "Luminosity", "nStar",
            "Surface_Density"]

    csoft = 0.001802390 / (0.6777) * 1e3

    data = {}
    intr_data = {}
    all_z_data = {}
    intr_all_z_data = {}

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
                "data/flares_sizes_all_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                "Total",
                                                                orientation),
                "r")
        except OSError as e:
            print(e)
            continue

        data.setdefault(snap, {})
        intr_data.setdefault(snap, {})

        for f in filters:

            print(reg, snap, f)
            all_z_data.setdefault(f, {})
            intr_all_z_data.setdefault(f, {})
            data[snap].setdefault(f, {})
            intr_data[snap].setdefault(f, {})

            for key in keys:
                data[snap][f].setdefault(key, []).extend(hdf[f][key][...])
                all_z_data[f].setdefault(key, []).extend(hdf[f][key][...])

            all_z_data[f].setdefault("Weight", []).extend(
                np.full(hdf[f]["Mass"][...].size, weights[int(reg)]))
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
            intr_all_z_data[f].setdefault("Inner_Surface_Density",
                                          []).extend(surf_dens)

            for key in keys:
                intr_data[snap][f].setdefault(key, []).extend(hdf[f][key][...])
                intr_all_z_data[f].setdefault(key, []).extend(hdf[f][key][...])

        hdf.close()

    for snap in snaps:

        for f in filters:

            print(snap, f)

            for key in data[snap][f].keys():
                data[snap][f][key] = np.array(data[snap][f][key])

            for key in intr_data[snap][f].keys():
                intr_data[snap][f][key] = np.array(
                    intr_data[snap][f][key])

            okinds = intr_data[snap][f]["nStar"] > 75

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

        print(f)

        for key in all_z_data[f].keys():
            all_z_data[f][key] = np.array(all_z_data[f][key])

        for key in intr_all_z_data[f].keys():
            intr_all_z_data[f][key] = np.array(
                intr_all_z_data[f][key])

        okinds = intr_all_z_data[f]["nStar"] > 75

        all_z_data[f]["okinds"] = okinds
        intr_all_z_data[f]["okinds"] = okinds

        compact_pop = np.array(
            intr_all_z_data[f]["Inner_Surface_Density"]) >= 10 ** 29
        diffuse_pop = np.array(
            intr_all_z_data[f]["Inner_Surface_Density"]) < 10 ** 29

        all_z_data[f]["Compact_Population"] = compact_pop
        all_z_data[f]["Diffuse_Population"] = diffuse_pop
        intr_all_z_data[f]["Compact_Population"] = compact_pop
        intr_all_z_data[f]["Diffuse_Population"] = diffuse_pop

    for f in filters:
        print("---------------------------", f,
              "---------------------------")
        mass_lumin(all_z_data[f]["Mass"],
                   all_z_data[f]["Luminosity"],
                   all_z_data[f]["okinds"],
                   all_z_data[f]["Diffuse_Population"],
                   all_z_data[f]["Compact_Population"],
                   all_z_data[f]["Weight"],
                   f, orientation, "Total", "default")
        mass_lumin(intr_all_z_data[f]["Mass"],
                   intr_all_z_data[f]["Luminosity"],
                   intr_all_z_data[f]["okinds"],
                   intr_all_z_data[f]["Diffuse_Population"],
                   intr_all_z_data[f]["Compact_Population"],
                   all_z_data[f]["Weight"],
                   f, orientation, "Intrinsic", "default")
