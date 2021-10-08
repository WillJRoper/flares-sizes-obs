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

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0])
    axright = fig.add_subplot(gs[1, 1])
    try:
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

    lumin_bins = np.logspace(27.2, 30.5, 50)
    Hbot2_all, bin_edges = np.histogram(lumins, bins=lumin_bins)
    Hbot2, bin_edges = np.histogram(lumins[nokinds], bins=lumin_bins)
    lbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axright.plot(lbin_cents, Hbot2_all, color="k", alpha=0.7)
    axright.plot(lbin_cents, Hbot2, color="k")

    mass_bins = np.logspace(8, 11, 50)
    Htop2_all, bin_edges = np.histogram(mass, bins=mass_bins)
    Htop2, bin_edges = np.histogram(mass[nokinds], bins=mass_bins)
    mbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axtop.plot(mbin_cents, Htop2_all, color="k", alpha=0.7)
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

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel(r"$L_{" + f.split(".")[-1] + "}/$ [erg $/$ s $/$ Hz]")
    ax.set_xlabel('$M_\star/ M_\odot$')

    ax.tick_params(axis='both', which='minor', bottom=True, left=True)

    ax.set_xlim(10 ** 8, 10 ** 11)
    axtop.set_xlim(10 ** 8, 10 ** 11)
    ax.set_ylim(10 ** 27.2, 10 ** 30.5)
    axright.set_ylim(10 ** 27.2, 10 ** 30.5)

    fig.savefig(
        'plots/' + str(z) + '/MassLumin_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + '.png',
        bbox_inches='tight')

    plt.close(fig)
