#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns

sns.set_context("paper")
sns.set_style('whitegrid')


def lum_comp(f, snap, lums, lums_pix, w, com_comp, diff_comp, com_ncomp,
              diff_ncomp, weight_norm, orientation, Type, extinction):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(lums[diff_ncomp], lums_pix[diff_ncomp],
                         C=w[diff_ncomp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', extent=(-1.1, 1.3, -1.1, 1.3),
                         alpha=0.2)
        cbar = ax.hexbin(lums[com_ncomp], lums_pix[com_ncomp],
                         C=w[com_ncomp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3),
                         alpha=0.2)
        cbar = ax.hexbin(lums[diff_comp], lums_pix[diff_comp],
                         C=w[diff_comp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', extent=(-1.1, 1.3, -1.1, 1.3))
        cbar = ax.hexbin(lums[com_comp], lums_pix[com_comp],
                         C=w[com_comp], gridsize=50, mincnt=1,
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
    ax.set_xlabel(r"$L_{" + f.split(".")[-1] + ", \mathrm{part}}/$ [erg $/$ s $/$ Hz]")
    ax.set_ylabel(r"$L_{" + f.split(".")[-1] + ", \mathrm{pix}}/$ [erg $/$ s $/$ Hz]")

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    plt.axis('scaled')

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)

    fig.savefig(
        'plots/' + str(z) + '/ComparisonLuminosity_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".png",
        bbox_inches='tight')
