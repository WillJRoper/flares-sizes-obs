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
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck13 as cosmo
import flare.photom as photconv
from flare import plt as flareplt


# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE

def m_to_M(m, cosmo, z):
    flux = photconv.m_to_flux(m)
    lum = photconv.flux_to_L(flux, cosmo, z)
    M = photconv.lum_to_M(lum)
    return M


def M_to_m(M, cosmo, z):
    lum = photconv.M_to_lum(M)
    flux = photconv.lum_to_flux(lum, cosmo, z)
    m = photconv.flux_to_m(flux)
    return m


def size_lumin_intrinsic(hlrs, lumins, w, com_comp, diff_comp, com_ncomp, diff_ncomp, f, snap,
                         mtype, orientation, Type, extinction, weight_norm, extent):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)
    print("Snap =", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    if w.size <= 1:
        return

    # lbins = np.logspace(26.8, 31.2, 40)
    # hbins = np.logspace(-1.5, 1.5, 40)
    # H, xbins, ybins = np.histogram2d(lumins[diff_ncomp], hlrs[okinds2],
    #                                  bins=(lbins, hbins), weights=w[okinds2])
    #
    # # Resample your data grid by a factor of 3 using cubic spline interpolation.
    # H = scipy.ndimage.zoom(H, 3)
    #
    # try:
    #     percentiles = [np.percentile(H[H > 0], 80),
    #                    np.percentile(H[H > 0], 90),
    #                    np.percentile(H[H > 0], 95),
    #                    np.percentile(H[H > 0], 99)]
    # except IndexError as e:
    #     print(e)
    #     return
    #
    # lbins = np.logspace(26.8, 31.2, H.shape[0] + 1)
    # hbins = np.logspace(-1.5, 1.5, H.shape[0] + 1)
    #
    # xbin_cents = (lbins[1:] + lbins[:-1]) / 2
    # ybin_cents = (hbins[1:] + hbins[:-1]) / 2
    #
    # XX, YY = np.meshgrid(xbin_cents, ybin_cents)

    fig = plt.figure(figsize=(3.5, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[10, 1, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    cax2 = fig.add_subplot(gs[0, 2])
    try:
        # cbar = ax.hexbin(lumins[diff_ncomp], hlrs[diff_ncomp],
        #                  C=w[diff_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='Greys',
        #                  extent=(26.8, 31.2, -1.5, 1.5), alpha=0.2)
        # cbar = ax.hexbin(lumins[com_ncomp], hlrs[com_ncomp],
        #                  C=w[com_ncomp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
        #                  xscale='log', yscale='log',
        #                  norm=weight_norm, linewidths=0.2,
        #                  cmap='viridis',
        #                  extent=(26.8, 31.2, -1.5, 1.5), alpha=0.2)
        im1 = ax.hexbin(lumins[diff_comp], hlrs[diff_comp],
                         C=w[diff_comp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys', extent=[extent[2], extent[3], extent[0],
                                               extent[1]])
        im2 = ax.hexbin(lumins[com_comp], hlrs[com_comp],
                         C=w[com_comp], gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=[extent[2], extent[3],
                                                 extent[0], extent[1]])
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)

    # Label axes
    ax.set_xlabel(r"$L_{\mathrm{" + f.split(".")[-1]
                  + "}}/$ [erg $/$ s $/$ Hz]")
    ax.set_ylabel('$R/ [pkpc]$')
    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(10 ** extent[2], 10 ** extent[3])
    ax.set_ylim(10 ** extent[0], 10 ** extent[1])

    cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=plt.get_cmap("Greys"),
                                    norm=weight_norm)
    cb1.ax.yaxis.set_ticks([])
    cb1 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.get_cmap("viridis"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")

    fig.savefig(
        'plots/' + str(z) + '/HalfLightRadius_' + mtype + "_" + f + '_' + str(
            z) + '_'
        + orientation + "_Intrinsic_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)
