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
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
import astropy.units as u
from FLARE.photom import lum_to_M, M_to_lum, lum_to_flux, m_to_flux
import FLARE.photom as photconv
import h5py
import sys
import pandas as pd
import utilities as util

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


# Set orientation
orientation = sys.argv[1]

# Define luminosity and mathrm{dust} model types
extinction = 'default'

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

# Define filter
filters = ('FAKE.TH.FUV', )

csoft = 0.001802390 / (0.6777) * 1e3

nlim = 700

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
weight_dict = {}

intr_hlr_dict = {}
intr_hlr_app_dict = {}
intr_hlr_pix_dict = {}
intr_lumin_dict = {}
intr_weight_dict = {}

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

for reg, snap in reg_snaps:

    hdf = h5py.File("data/flares_sizes_{}_{}_10th.hdf5".format(reg, snap), "r")
    type_group = hdf["Total"]
    orientation_group = type_group[orientation]

    hlr_dict.setdefault(snap, {})
    hlr_app_dict.setdefault(snap, {})
    hlr_pix_dict.setdefault(snap, {})
    lumin_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hlr_app_dict[snap].setdefault(f, [])
        hlr_pix_dict[snap].setdefault(f, [])
        lumin_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = orientation_group[f]["Mass"][...]
        okinds = orientation_group[f]["nStar"][...] > nlim

        print(reg, snap, f, masses[okinds].size)

        hlr_dict[snap][f].extend(orientation_group[f]["HLR_0.5"][...][okinds])
        hlr_app_dict[snap][f].extend(
            orientation_group[f]["HLR_Aperture_0.5"][...][okinds])
        hlr_pix_dict[snap][f].extend(
            orientation_group[f]["HLR_Pixel_0.5"][...][okinds])
        lumin_dict[snap][f].extend(
            orientation_group[f]["Luminosity"][...][okinds])
        weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))

    type_group = hdf["Intrinsic"]
    orientation_group = type_group[orientation]

    intr_hlr_dict.setdefault(snap, {})
    intr_hlr_app_dict.setdefault(snap, {})
    intr_hlr_pix_dict.setdefault(snap, {})
    intr_lumin_dict.setdefault(snap, {})
    intr_weight_dict.setdefault(snap, {})

    for f in filters:
        intr_hlr_dict[snap].setdefault(f, [])
        intr_hlr_app_dict[snap].setdefault(f, [])
        intr_hlr_pix_dict[snap].setdefault(f, [])
        intr_lumin_dict[snap].setdefault(f, [])
        intr_weight_dict[snap].setdefault(f, [])

        masses = orientation_group[f]["Mass"][...]
        okinds = orientation_group[f]["nStar"][...] > nlim

        print(reg, snap, f, masses[okinds].size)

        intr_hlr_dict[snap][f].extend(orientation_group[f]["HLR_0.5"][...][okinds])
        intr_hlr_app_dict[snap][f].extend(
            orientation_group[f]["HLR_Aperture_0.5"][...][okinds])
        intr_hlr_pix_dict[snap][f].extend(
            orientation_group[f]["HLR_Pixel_0.5"][...][okinds])
        intr_lumin_dict[snap][f].extend(
            orientation_group[f]["Luminosity"][...][okinds])
        intr_weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))


    hdf.close()

for f in filters:

    fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                             np.log10(M_to_lum(-18)),
                             1000)

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Filter =", f)

    for snap in snaps:

        legend_elements = []

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        intr_hlrs = np.array(intr_hlr_dict[snap][f])
        intr_lumins = np.array(intr_lumin_dict[snap][f])
        w = np.array(weight_dict[snap][f])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax1.grid(False)
        ax.loglog()
        ax1.loglog()
        try:
            ax.scatter(intr_lumins, intr_hlrs, color="k", marker="D",
                       alpha=0.6)
            ax1.hexbin(lumins, hlrs * cosmo.arcsec_per_kpc_proper(z).value,
                       gridsize=50, mincnt=1, xscale='log',
                       yscale='log', norm=LogNorm(), linewidths=0.2,
                       cmap='viridis', alpha=0)
            sinds = np.argsort(lumins / intr_lumins)
            # for intr_l, l, intr_r, r in zip(intr_lumins[sinds][-10:],
            #                                 lumins[sinds][-10:],
            #                                 intr_hlrs[sinds][-10:],
            #                                 hlrs[sinds][-10:]):
            #     ax.plot((intr_l, l), (intr_r, r), linestyle="-", color="k",
            #             marker=None, alpha=0.6)
            im = ax.scatter(lumins[sinds], hlrs[sinds],
                            c=np.log10(lumins[sinds] / intr_lumins[sinds]),
                            marker="D",
                            cmap="viridis")
        except ValueError as e:
            print(e)
            continue

        ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
        cbar = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
        cbaxes.xaxis.set_ticks_position("top")
        cbar.ax.set_xlabel("$\log_{10}(L_{\mathrm{ex}}/L_{\mathrm{int}})$",
                           labelpad=-40)

        ax.set_xlim(10**27.9, 10**31.1)
        ax.set_ylim(10**-1.2, 10**1.4)

        fig.savefig(
            'plots/' + str(z) + '/HalfLightRadius_dust_effects__' + f + '_' + str(
                z) + '_'
            + orientation + "_" + extinction + "_"
            + '%d.png' % nlim,
            bbox_inches='tight')

        plt.close(fig)

        legend_elements = []

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_app_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        intr_hlrs = np.array(intr_hlr_app_dict[snap][f])
        intr_lumins = np.array(intr_lumin_dict[snap][f])
        w = np.array(weight_dict[snap][f])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax1.grid(False)
        ax.loglog()
        ax1.loglog()
        try:
            ax.scatter(intr_lumins, intr_hlrs, color="k", marker="D",
                       alpha=0.6)
            for intr_l, l, intr_r, r in zip(intr_lumins, lumins,
                                            intr_hlrs, hlrs):
                ax.plot((intr_l, l), (intr_r, r), linestyle="-", color="k",
                        alpha=0.3)
            im = ax.scatter(lumins, hlrs,
                            c=lum_to_M(intr_lumins) - lum_to_M(lumins),
                            marker="D",
                            alpha=0.7,
                            cmap="viridis")
        except ValueError as e:
            print(e)
            continue

        ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
        cbar = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
        cbaxes.xaxis.set_ticks_position("top")
        cbar.ax.set_xlabel("$A$", labelpad=-30)

        ax.set_xlim(10**27.9, 10**31.1)
        ax.set_ylim(10**-1.2, 10**1.4)

        fig.savefig('plots/' + str(z) + '/HalfLightRadius_dust_effects_Aperture_'
                    + f + '_' + str(z) + '_' + orientation
                    + "_" + extinction + "_"
                    + '%d.png' % nlim,
                    bbox_inches='tight')

        plt.close(fig)

        legend_elements = []

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_pix_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        intr_hlrs = np.array(intr_hlr_pix_dict[snap][f])
        intr_lumins = np.array(intr_lumin_dict[snap][f])
        w = np.array(weight_dict[snap][f])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = ax.twinx()
        ax1.grid(False)
        ax.loglog()
        ax1.loglog()
        try:
            ax.scatter(intr_lumins, intr_hlrs, color="k", marker="D", 
                       alpha=0.6)
            for intr_l, l, intr_r, r in zip(intr_lumins, lumins, 
                                            intr_hlrs, hlrs):
                ax.plot((intr_l, l), (intr_r, r), linestyle="-", color="k",
                        alpha=0.3)
            im = ax.scatter(lumins, hlrs, 
                            c=lum_to_M(intr_lumins) - lum_to_M(lumins), 
                            marker="D",
                            alpha=0.7, 
                            cmap="viridis")
        except ValueError as e:
            print(e)
            continue

        ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
        cbar = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
        cbaxes.xaxis.set_ticks_position("top")
        cbar.ax.set_xlabel("$A$", labelpad=-30)

        ax.set_xlim(10**27.9, 10**31.1)
        ax.set_ylim(10**-1.2, 10**1.4)

        fig.savefig('plots/' + str(z) + '/HalfLightRadius_dust_effects_Pixel_'
                    + f + '_' + str(z) + '_' + orientation
                    + "_" + extinction + "_"
                    + '%d.png' % nlim,
                    bbox_inches='tight')

        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax1 = ax.twinx()
        # ax1.grid(False)
        ax.loglog()
        # ax1.loglog()
        try:
            extinc = lum_to_M(intr_lumins) - lum_to_M(lumins)
            sinds = np.argsort(extinc)[::-1]
            im = ax.scatter(intr_hlrs[sinds], hlrs[sinds],
                            c=extinc[sinds],
                            marker="D",
                            cmap="viridis")
        except ValueError as e:
            print(e)
            continue

        # ax1.set_ylabel('$R_{1/2,\mathrm{dust}}/ [arcsecond]$')

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel('$R_{1/2,\mathrm{intrinsic}}/ [pkpc]$')
        ax.set_ylabel('$R_{1/2, mathrm{dust}}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
        cbar = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
        cbaxes.xaxis.set_ticks_position("top")
        cbar.ax.set_xlabel("$A$", labelpad=-50)

        ax.set_xlim(10**-1.2, 10**1.4)
        ax.set_ylim(10**-1.2, 10**1.4)

        fig.savefig('plots/' + str(z) + '/HalfLightRadius_dust_effects_1to1'
                                        '_Pixel_'
                    + f + '_' + str(z) + '_' + orientation
                    + "_" + extinction + "_"
                    + '%d.png' % nlim,
                    bbox_inches='tight')

        plt.close(fig)

        hlrs = np.array(hlr_dict[snap][f])
        intr_hlrs = np.array(intr_hlr_dict[snap][f])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax1 = ax.twinx()
        # ax1.grid(False)
        ax.loglog()
        # ax1.loglog()
        try:
            extinc = lum_to_M(intr_lumins) - lum_to_M(lumins)
            sinds = np.argsort(extinc)[::-1]
            im = ax.scatter(intr_hlrs[sinds], hlrs[sinds],
                            c=extinc[sinds],
                            marker="D",
                            cmap="viridis")
        except ValueError as e:
            print(e)
            continue

        # ax1.set_ylabel('$R_{1/2,\mathrm{dust}}/ [arcsecond]$')

        ax.text(0.95, 0.05, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel('$R_{1/2,\mathrm{intrinsic}}/ [pkpc]$')
        ax.set_ylabel('$R_{1/2, mathrm{dust}}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        cbaxes = ax.inset_axes([0.0, 1.0, 1.0, 0.04])
        cbar = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
        cbaxes.xaxis.set_ticks_position("top")
        cbar.ax.set_xlabel("$A$", labelpad=-50)

        ax.set_xlim(10**-1.2, 10**1.4)
        ax.set_ylim(10**-1.2, 10**1.4)

        fig.savefig('plots/' + str(z) + '/HalfLightRadius_dust_effects_1to1_'
                    + f + '_' + str(z) + '_' + orientation
                    + "_" + extinction + "_"
                    + '%d.png' % nlim,
                    bbox_inches='tight')

        plt.close(fig)
