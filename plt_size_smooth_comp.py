#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
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


def size_comp_smooth(f, snap, hlrs_pix, hlrs_pix_nosmooth, w, com_comp,
                     diff_comp, com_ncomp,
                     diff_ncomp, weight_norm,
                     orientation, Type, extinction):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    try:
        # cbar = ax.hexbin(hlrs_pix[diff_ncomp], hlrs_pix_nosmooth[diff_ncomp],
        #                  C=w[diff_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='Greys', extent=(-1.1, 1.3, -1.1, 1.3),
        #                  alpha=0.2)
        # cbar = ax.hexbin(hlrs_pix[com_ncomp], hlrs_pix_nosmooth[com_ncomp],
        #                  C=w[com_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3),
        #                  alpha=0.2)
        cbar = ax.hexbin(hlrs_pix, hlrs_pix_nosmooth,
                         C=w, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3))
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** -1.1, 10 ** 1.3], [10 ** -1.1, 10 ** 1.3],
            color='k', linestyle="--")

    # Label axes
    ax.set_ylabel('$R_{\mathrm{No Smooth}}/ [pkpc]$')
    ax.set_xlabel('$R_{\mathrm{Smooth}}/ [pkpc]$')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    plt.axis('scaled')

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.1)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("viridis"), norm=weight_norm, orientation="horizontal")
    cb1.set_label("$\sum w_{i}$")
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')

    fig.savefig(
        'plots/' + str(z) + '/ComparisonHalfLightRadiusSmoothing_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')
