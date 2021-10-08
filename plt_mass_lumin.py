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

def mass_lumin(mass, lumins, okinds1, okinds2, nstar, w,
               f, snap, orientation, Type, extinction):

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Snapshot =", snap)
    print("Filter =", f)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    nokinds1 = nstar >= 100
    nokinds2 = nstar < 100

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

    Htop2, bin_edges = np.histogram(lumins, )

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
        + orientation + '_' + Type + "_" + extinction + '.png',
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
