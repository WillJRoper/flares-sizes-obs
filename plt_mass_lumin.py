#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

mpl.use('Agg')
warnings.filterwarnings('ignore')
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from flare import plt as flareplt


# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def mass_lumin(mass, lumins, com_comp, diff_comp, com_ncomp, diff_ncomp, okinds, w,
               f, snap, orientation, Type, extinction, comp_l, comp_m, weight_norm, extent):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Snapshot =", snap)
    print("Filter =", f)

    complete = np.logical_or(com_comp, diff_comp)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

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
        cbar = ax.hexbin(mass[diff_ncomp], lumins[diff_ncomp],
                         gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)), C=w[diff_ncomp],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', alpha=0.2,
                         extent=extent)
        cbar = ax.hexbin(mass[com_ncomp], lumins[com_ncomp],
                         gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)), C=w[com_ncomp],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', alpha=0.2,
                         extent=extent)
        cbar = ax.hexbin(mass[diff_comp], lumins[diff_comp],
                         gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)), C=w[diff_comp],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys',
                         extent=extent)
        cbar = ax.hexbin(mass[com_comp], lumins[com_comp],
                         gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)), C=w[com_comp],
                         reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis',
                         extent=extent)
    except ValueError as e:
        print(e)

    lumin_bins = np.logspace(extent[2], extent[3], 50)
    Hbot2_all, bin_edges = np.histogram(lumins, bins=lumin_bins)
    Hbot2, bin_edges = np.histogram(lumins[okinds], bins=lumin_bins)
    lbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axright.plot(Hbot2_all, lbin_cents, color="k", alpha=0.4)
    axright.plot(Hbot2, lbin_cents, color="k")
    axright.axhline(comp_l, linestyle="--", alpha=0.6, color="k")
    ax.axhline(comp_l, linestyle="--", alpha=0.6, color="k")

    mass_bins = np.logspace(extent[0], extent[1], 50)
    Htop2_all, bin_edges = np.histogram(mass, bins=mass_bins)
    Htop2, bin_edges = np.histogram(mass[okinds], bins=mass_bins)
    mbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axtop.plot(mbin_cents, Htop2_all, color="k", alpha=0.4)
    axtop.plot(mbin_cents, Htop2, color="k")
    axtop.axvline(comp_m, linestyle="--", alpha=0.6, color="k")
    ax.axvline(comp_m, linestyle="--", alpha=0.6, color="k")

    # Remove axis labels and ticks
    axtop.tick_params(axis='x', top=False, bottom=False,
                      labeltop=False, labelbottom=False)
    # axtop.tick_params(axis='y', left=False, right=False,
    #                   labelleft=False, labelright=False)
    # axright.tick_params(axis='x', top=False, bottom=False,
    #                     labeltop=False, labelbottom=False)
    axright.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

    axtop.spines['top'].set_visible(False)
    axtop.spines['right'].set_visible(False)
    # axtop.spines['left'].set_visible(False)

    # axright.spines['bottom'].set_visible(False)
    axright.spines['top'].set_visible(False)
    axright.spines['right'].set_visible(False)

    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap("Greys"), norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")

    ax2 = fig.add_axes([0.1, 0.95, 0.8, 0.03])
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap("viridis"), norm=weight_norm, orientation="horizontal")
    cb1.set_label("$\sum w_{i}$")
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')

    # ax.text(0.95, 0.05, f'$z={z}$',
    #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
    #                   ec="k", lw=1, alpha=0.8),
    #         transform=ax.transAxes, horizontalalignment='right',
    #         fontsize=8)

    # Label axes
    ax.set_ylabel(r"$L_{" + f.split(".")[-1] + "}/$ [erg $/$ s $/$ Hz]")
    ax.set_xlabel('$M_\star/ M_\odot$')

    axtop.set_ylabel("$N$")
    axright.set_xlabel("$N$")

    ax.tick_params(axis='both', which='both', bottom=True, left=True)

    axtop.tick_params(axis='y', which='both', left=True)
    axright.tick_params(axis='x', which='both', bottom=True)

    ax.set_xlim(10 ** extent[0], 10 ** extent[1])
    axtop.set_xlim(10 ** extent[0], 10 ** extent[1])
    ax.set_ylim(10 ** extent[2], 10 ** extent[3])
    axright.set_ylim(10 ** extent[2], 10 ** extent[3])

    axright.set_xlim(1, 10 ** 4)
    axtop.set_ylim(1, 10 ** 4)

    fig.savefig(
        'plots/' + str(z) + '/MassLumin_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + '.pdf',
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

            okinds = intr_data[snap][f]["nStar"] > 100

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
            mass_lumin(data[snap][f]["Mass"],
                       data[snap][f]["Luminosity"],
                       data[snap][f]["okinds"],
                       data[snap][f]["Diffuse_Population"],
                       data[snap][f]["Compact_Population"],
                       data[snap][f]["Weight"],
                       f, snap, orientation, "Total", "default")
            mass_lumin(intr_data[snap][f]["Mass"],
                       intr_data[snap][f]["Luminosity"],
                       intr_data[snap][f]["okinds"],
                       intr_data[snap][f]["Diffuse_Population"],
                       intr_data[snap][f]["Compact_Population"],
                       data[snap][f]["Weight"],
                       f, snap, orientation, "Intrinsic", "default")
