#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

from flare import plt


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
        #                  mincnt=1,
        #                  C=w[diff_ncomp], reduce_C_function=np.sum,
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2, cmap='Greys',
        #                  alpha=0.2)
        # ax.hexbin(hdrs[com_ncomp], hlrs[com_ncomp], gridsize=50,
        #           mincnt=1,
        #           C=w[com_ncomp],
        #           reduce_C_function=np.sum, xscale='log', yscale='log',
        #           norm=weight_norm, linewidths=0.2, cmap='viridis', alpha=0.2)

        cbar = ax.hexbin(hdrs[diff_comp], hlrs[diff_comp], gridsize=50,
                         mincnt=1,
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys')
        ax.hexbin(hdrs[com_comp], hlrs[com_comp], gridsize=50,
                  mincnt=1,
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
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    plt.axis('scaled')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)

    # fig.colorbar(cbar)

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
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
        cbar = ax.hexbin(hdrs[diff_ncomp], ratio[diff_ncomp], gridsize=50,
                         mincnt=1,
                         C=w[diff_ncomp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys',
                         alpha=0.2)
        ax.hexbin(hdrs[com_ncomp], ratio[com_ncomp], gridsize=50,
                  mincnt=1,
                  C=w[com_ncomp], reduce_C_function=np.sum,
                  xscale='log', yscale='log', norm=weight_norm,
                  linewidths=0.2, cmap='viridis', alpha=0.2)
        cbar = ax.hexbin(hdrs[diff_comp], ratio[diff_comp], gridsize=50,
                         mincnt=1,
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys')
        ax.hexbin(hdrs[com_comp], ratio[com_comp], gridsize=50,
                  mincnt=1,
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
    ax.set_ylabel("$R_{1/2," + f.split(".")[-1] + "}/ R_{1/2, dust}$")
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim([0.08, 50])

    plt.axis('scaled')

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_ratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight')

    plt.close(fig)

    # ==================================================================

    ratio = hlrs / hlrints

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
        cbar = ax.hexbin(hdrs[diff_ncomp], ratio[diff_ncomp], gridsize=50,
                         mincnt=1,
                         C=w[diff_ncomp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys',
                         alpha=0.2)
        ax.hexbin(hdrs[com_ncomp], ratio[com_ncomp], gridsize=50,
                  mincnt=1,
                  C=w[com_ncomp], reduce_C_function=np.sum,
                  xscale='log', yscale='log', norm=weight_norm,
                  linewidths=0.2, cmap='viridis', alpha=0.2)
        cbar = ax.hexbin(hdrs[diff_comp], ratio[diff_comp], gridsize=50,
                         mincnt=1,
                         C=w[diff_comp], reduce_C_function=np.sum,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2, cmap='Greys')
        ax.hexbin(hdrs[com_comp], ratio[com_comp], gridsize=50, mincnt=1,
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
    ax.tick_params(axis='both', which='both', bottom=True, left=True)

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_ylabel("$R_{1/2,"
                  + f.split(".")[-1]
                  + ", \mathrm{Attenuated}}/ R_{1/2,"
                  + f.split(".")[-1] + ", \mathrm{Intrinsic}}$")
    ax.set_xlabel('$R_{1/2, dust}/ [pkpc]$')

    ax.set_xlim(10 ** -1.1, 10 ** 1.7)
    ax.set_ylim([0.15, 50])

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_hlrratio_' + f + '_'
                + str(z) + '_' + Type + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight')

    plt.close(fig)
