#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

mpl.use('Agg')
warnings.filterwarnings('ignore')
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


def size_comp(f, snap, hlrs, hlrs_pix, w, com_comp, diff_comp, com_ncomp,
              diff_ncomp, weight_norm, orientation, Type, extinction, extent):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    fig = plt.figure(figsize=(3.5 + 1/20 * 3.5, 2 * 3.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])
    ax1.loglog()
    ax2.loglog()
    try:
        # cbar = ax.hexbin(hlrs[diff_ncomp], hlrs_pix[diff_ncomp],
        #                  C=w[diff_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='Greys', extent=extent,
        #                  alpha=0.2)
        # cbar = ax.hexbin(hlrs[com_ncomp], hlrs_pix[com_ncomp],
        #                  C=w[com_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='viridis', extent=extent,
        #                  alpha=0.2)
        cbar = ax1.hexbin(hlrs[diff_comp], hlrs_pix[diff_comp],
                         C=w[diff_comp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', extent=extent)
        ax1.axis('scaled')
        cbar = ax2.hexbin(hlrs[com_comp], hlrs_pix[com_comp],
                         C=w[com_comp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=extent)
        ax2.axis('scaled')
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)
        return

    ax1.plot([10 ** extent[0], 10 ** extent[1]],
            [10 ** extent[2], 10 ** extent[3]],
            color='k', linestyle="--")
    ax2.plot([10 ** extent[0], 10 ** extent[1]],
            [10 ** extent[2], 10 ** extent[3]],
            color='k', linestyle="--")

    # Label axes
    ax2.set_xlabel('$R_{\mathrm{part}}/ [pkpc]$')
    ax1.set_ylabel('$R_{\mathrm{pix}}/ [pkpc]$')
    ax2.set_ylabel('$R_{\mathrm{pix}}/ [pkpc]$')

    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='both', which='both', left=True, bottom=True)
    ax2.tick_params(axis='both', which='both', left=True, bottom=True)

    ax1.set_xlim(10 ** extent[0], 10 ** extent[1])
    ax1.set_ylim(10 ** extent[2], 10 ** extent[3])
    ax2.set_xlim(10 ** extent[0], 10 ** extent[1])
    ax2.set_ylim(10 ** extent[2], 10 ** extent[3])

    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.get_cmap("Greys"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")
    cb1.set_ticks([10**-3, 10**-2, 10**-1, 1])
    cb1 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.get_cmap("viridis"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")

    fig.savefig(
        'plots/' + str(z) + '/ComparisonHalfLightRadius_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')
