#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
import scipy
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


def fit(x, m, c):
    return 10**(m * np.log10(x) + c)


def hdr_comp(hdrs, hlrs, hlrints, w, com_comp, diff_comp, com_ncomp,
             diff_ncomp, f, orientation,
             snap, Type, extinction, weight_norm):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Filter =", f)
    print("Snapshot =", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # bins = np.logspace(np.log10(0.08), np.log10(20), 40)
    # H, xbins, ybins = np.histogram2d(hdrs[diff_comp], hlrs[diff_comp],
    #                                  bins=bins, weights=w[diff_comp])
    # 
    # # Resample your data grid by a factor of 3 using cubic spline interpolation.
    # H = scipy.ndimage.zoom(H, 3)
    # 
    # # percentiles = [np.min(w),
    # #                10**-3,
    # #                10**-1,
    # #                1, 2, 5]
    # 
    # try:
    #     percentiles = [np.percentile(H[H > 0], 50),
    #                    np.percentile(H[H > 0], 80),
    #                    np.percentile(H[H > 0], 90),
    #                    np.percentile(H[H > 0], 95),
    #                    np.percentile(H[H > 0], 99)]
    # except IndexError:
    #     return
    # 
    # bins = np.logspace(np.log10(0.08), np.log10(20), H.shape[0] + 1)
    # 
    # xbin_cents = (bins[1:] + bins[:-1]) / 2
    # ybin_cents = (bins[1:] + bins[:-1]) / 2
    # 
    # XX, YY = np.meshgrid(xbin_cents, ybin_cents)
    # 
    # print(percentiles)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    try:
        # cbar = ax.hexbin(hdrs[diff_ncomp], hlrs[diff_ncomp], gridsize=50,
        #                  mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  C=w[diff_ncomp], reduce_C_function=np.sum,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2, cmap='Greys',
        #                  alpha=0.2)
        # ax.hexbin(hdrs[com_ncomp], hlrs[com_ncomp], gridsize=50,
        #           mincnt=np.min(w) - (0.1 * np.min(w)),
        #           C=w[com_ncomp],
        #           reduce_C_function=np.sum, xscale='log', yscale='log',
        #           norm=weight_norm, linewidths=0.2, cmap='viridis', alpha=0.2)

        cbar = ax.hexbin(hdrs[diff_comp], hlrs[diff_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys')
        ax.hexbin(hdrs[com_comp], hlrs[com_comp], gridsize=50,
                  mincnt=np.min(w) - (0.1 * np.min(w)),
                  C=w[com_comp],
                  reduce_C_function=np.sum, xscale='log', yscale='log',
                  norm=weight_norm, linewidths=0.2, cmap='viridis')
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   locator=ticker.LogLocator(),
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)

    min = np.min((ax.get_xlim(), ax.get_ylim()))
    max = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.plot([min, max], [min, max], color='k', linestyle="--")

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ [pkpc]$")
    ax.set_xlabel('$R_{1/2, metal}/ [pkpc]$')

    plt.axis('scaled')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)

    # fig.colorbar(cbar)

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".pdf",
                bbox_inches='tight')

    plt.close(fig)

    ratio = hlrs / hdrs

    # bins = np.logspace(np.log10(0.08), np.log10(40), 40)
    # 
    # H, xbins, ybins = np.histogram2d(hdrs[diff_comp], ratio[diff_comp],
    #                                  bins=bins,
    #                                  weights=w[diff_comp])
    # 
    # # Resample your data grid by a factor of 3 using cubic spline interpolation.
    # H = scipy.ndimage.zoom(H, 3)
    # 
    # # percentiles = [np.min(w),
    # #                10**-3,
    # #                10**-1,
    # #                1, 2, 5]
    # 
    # try:
    #     percentiles = [np.percentile(H[H > 0], 50),
    #                    np.percentile(H[H > 0], 80),
    #                    np.percentile(H[H > 0], 90),
    #                    np.percentile(H[H > 0], 95),
    #                    np.percentile(H[H > 0], 99)]
    # except IndexError:
    #     return
    # 
    # bins = np.logspace(np.log10(0.08), np.log10(40), H.shape[0] + 1)
    # 
    # xbin_cents = (bins[1:] + bins[:-1]) / 2
    # ybin_cents = (bins[1:] + bins[:-1]) / 2
    # 
    # XX, YY = np.meshgrid(xbin_cents, ybin_cents)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog()
    try:
        # cbar = ax.hexbin(hdrs[diff_ncomp], ratio[diff_ncomp], gridsize=50,
        #                  mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  C=w[diff_ncomp], reduce_C_function=np.sum,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2, cmap='Greys',
        #                  alpha=0.2)
        # ax.hexbin(hdrs[com_ncomp], ratio[com_ncomp], gridsize=50,
        #           mincnt=np.min(w) - (0.1 * np.min(w)),
        #           C=w[com_ncomp], reduce_C_function=np.sum,
        #           xscale='log', yscale='log', norm=weight_norm,
        #           linewidths=0.2, cmap='viridis', alpha=0.2)
        cbar = ax.hexbin(hdrs[diff_comp], ratio[diff_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys')
        ax.hexbin(hdrs[com_comp], ratio[com_comp], gridsize=50,
                  mincnt=np.min(w) - (0.1 * np.min(w)),
                  C=w[com_comp], reduce_C_function=np.sum,
                  xscale='log', yscale='log', norm=weight_norm,
                  linewidths=0.2, cmap='viridis')
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)

    min = np.min((ax.get_xlim(), ax.get_ylim()))
    max = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.plot([min, max], [1, 1], color='k', linestyle="--")

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ R_{1/2, metal}$")
    ax.set_xlabel('$R_{1/2, metal}/ [pkpc]$')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim([0.08, 50])

    plt.axis('scaled')

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_ratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".pdf",
                bbox_inches='tight')

    plt.close(fig)

    # ==================================================================

    ratio = hlrs / hlrints

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[40, 1])
    gs.update(wspace=-2, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[1, 1])
    ax1.loglog()
    ax2.loglog()
    try:
        # cbar = ax.hexbin(hdrs[diff_ncomp], ratio[diff_ncomp], gridsize=50,
        #                  mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  C=w[diff_ncomp], reduce_C_function=np.sum,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2, cmap='Greys',
        #                  alpha=0.2)
        # ax.hexbin(hdrs[com_ncomp], ratio[com_ncomp], gridsize=50,
        #           mincnt=np.min(w) - (0.1 * np.min(w)),
        #           C=w[com_ncomp], reduce_C_function=np.sum,
        #           xscale='log', yscale='log', norm=weight_norm,
        #           linewidths=0.2, cmap='viridis', alpha=0.2)
        ax1.hexbin(hdrs[diff_comp], ratio[diff_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys',
                         extent=[-1.1, 1.3, np.log10(0.2), np.log10(50)])
        ax1.axis('scaled')
        ax2.hexbin(hdrs[com_comp], ratio[com_comp], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[com_comp], reduce_C_function=np.sum,
                   xscale='log', yscale='log', norm=weight_norm,
                   linewidths=0.2, cmap='viridis', extent=[-1.1, 1.3,
                                                           np.log10(0.2),
                                                           np.log10(80)])
        ax2.axis('scaled')
    except ValueError as e:
        print(e)

    for ax in [ax1, ax2]:
        ax.plot([10**-1.1, 10**1.3], [1, 1], color='k', linestyle="--")
        ax.set_xlim(10 ** -1.1, 10 ** 1.3)
        ax.set_ylim([0.2, 80])

    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='both', which='both', bottom=True, left=True)
    ax2.tick_params(axis='x', which='both', bottom=True)

    # ax.text(0.95, 0.05, f'$z={z}$',
    #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
    #                   ec="k", lw=1, alpha=0.8),
    #         transform=ax.transAxes, horizontalalignment='right',
    #         fontsize=8)

    # Label axes
    ax1.set_ylabel("$R_{1/2,"
                  + f.split(".")[-1]
                  + ", \mathrm{Att}}/ R_{1/2,"
                  + f.split(".")[-1] + ", \mathrm{Int}}$")
    ax2.set_ylabel("$R_{1/2,"
                  + f.split(".")[-1]
                  + ", \mathrm{Att}}/ R_{1/2,"
                  + f.split(".")[-1] + ", \mathrm{Int}}$")
    ax2.set_xlabel('$R_{1/2, metal}/ [pkpc]$')

    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.get_cmap("Greys"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")
    cb1.set_ticks([10**-3, 10**-2, 10**-1, 1])
    cb1 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.get_cmap("viridis"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_hlrratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".pdf",
                bbox_inches='tight')

    plt.close(fig)
