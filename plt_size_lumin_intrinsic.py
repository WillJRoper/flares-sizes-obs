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
from astropy.cosmology import Planck13 as cosmo
import flare.photom as photconv
import cmasher as cmr
import scipy.ndimage

sns.set_context("paper")
sns.set_style('whitegrid')


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
                         mtype, orientation, Type, extinction, weight_norm):
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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = ax.twinx()
    ax1.grid(False)
    try:
        cbar = ax.hexbin(lumins[diff_ncomp], hlrs[diff_ncomp],
                         C=w[diff_ncomp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys',
                         extent=(26.8, 31.2, -1.5, 1.5), alpha=0.2)
        cbar = ax.hexbin(lumins[com_ncomp], hlrs[com_ncomp],
                         C=w[com_ncomp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis',
                         extent=(26.8, 31.2, -1.5, 1.5), alpha=0.2)
        cbar = ax.hexbin(lumins[diff_comp], hlrs[diff_comp],
                         C=w[diff_comp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys',
                         extent=(26.8, 31.2, -1.5, 1.5))
        cbar = ax.hexbin(lumins[com_comp], hlrs[com_comp],
                         C=w[com_comp], gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis',
                         extent=(26.8, 31.2, -1.5, 1.5))
        # cbar = ax.contour(XX, YY, H.T, levels=percentiles,
        #                   norm=weight_norm, cmap=cmr.bubblegum_r,
        #                   linewidth=2)
    except ValueError as e:
        print(e)

    try:
        ax1.hexbin(lumins,
                   hlrs * cosmo.arcsec_per_kpc_proper(z).value,
                   gridsize=50, mincnt=1, C=w,
                   reduce_C_function=np.sum, xscale='log',
                   yscale='log', norm=weight_norm, linewidths=0.2,
                   cmap='viridis', alpha=0,
                   extent=(26.8, 31.2,
                           np.log10(10 ** -1.5
                                    * cosmo.arcsec_per_kpc_proper(z).value),
                           np.log10(10 ** 1.5
                                    * cosmo.arcsec_per_kpc_proper(z).value)))
    except ValueError as e:
        print(e)

    ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

    # Label axes
    ax.set_xlabel(r"$L_{" + f.split(".")[-1]
                  + "}/$ [erg $/$ s $/$ Hz]")
    ax.set_ylabel('$R_{1/2}/ [pkpc]$')

    ax1.set_xlim(10 ** 26.5, 10 ** 31.2)
    ax.set_xlim(10 ** 26.5, 10 ** 31.2)
    ax.set_ylim(10 ** -1.5, 10 ** 1.5)
    ax1.set_ylim(10 ** -1.5 * cosmo.arcsec_per_kpc_proper(z).value,
                 10 ** 1.5 * cosmo.arcsec_per_kpc_proper(z).value)

    ax.tick_params(axis='x', which='minor', bottom=True)

    fig.savefig(
        'plots/' + str(z) + '/HalfLightRadius_' + mtype + "_" + f + '_' + str(
            z) + '_'
        + orientation + "_Intrinsic_" + extinction + ".png",
        bbox_inches='tight')

    plt.close(fig)
