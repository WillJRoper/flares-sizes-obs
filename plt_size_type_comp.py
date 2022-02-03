#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        # cbar = ax.hexbin(hlrs[diff_ncomp], hlrs_pix[diff_ncomp],
        #                  C=w[diff_ncomp], gridsize=50, mincnt=1,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='Greys', extent=extent,
        #                  alpha=0.2)
        # cbar = ax.hexbin(hlrs[com_ncomp], hlrs_pix[com_ncomp],
        #                  C=w[com_ncomp], gridsize=50, mincnt=1,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='viridis', extent=extent,
        #                  alpha=0.2)
        cbar = ax.hexbin(hlrs[diff_comp], hlrs_pix[diff_comp],
                         C=w[diff_comp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', extent=extent)
        cbar = ax.hexbin(hlrs[com_comp], hlrs_pix[com_comp],
                         C=w[com_comp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=extent)
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** extent[0], 10 ** extent[1]],
            [10 ** extent[2], 10 ** extent[3]],
            color='k', linestyle="--")

    # Label axes
    ax.set_xlabel('$R_{1/2, \mathrm{part}}/ [pkpc]$')
    ax.set_ylabel('$R_{1/2, \mathrm{pix}}/ [pkpc]$')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    plt.axis('scaled')

    ax.set_xlim(10 ** extent[0], 10 ** extent[1])
    ax.set_ylim(10 ** extent[2], 10 ** extent[3])

    fig.savefig(
        'plots/' + str(z) + '/ComparisonHalfLightRadius_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')
