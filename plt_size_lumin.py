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
sns.set_style('white')

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_up_params = {'beta': {6: 0.08, 7: 0.08,
                           8: 0.28, 9: 1.01},
                  'r_0': {6: 0.2, 7: 0.2,
                          8: 5.28, 9: 367.64}}
kawa_low_params = {'beta': {6: 0.09, 7: 0.09,
                            8: 0.78, 9: 0.27},
                   'r_0': {6: 0.15, 7: 0.15,
                           8: 0.26, 9: 0.74}}
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b


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


def kawa_fit_err(y, l, ro, b, ro_err, b_err, uplow="up"):
    ro_term = ro_err * (l / M_to_lum(-21)) ** b
    beta_term = b_err * ro * (l / M_to_lum(-21)) ** b \
                * np.log(l / M_to_lum(-21))

    if uplow == "up":
        return y + np.sqrt(ro_term ** 2 + beta_term ** 2)
    else:
        return y - np.sqrt(ro_term ** 2 + beta_term ** 2)


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):
    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 15)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median',
                                                 bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
            label=lab)


def r_from_surf_den(lum, s_den):

    return np.sqrt(lum / (s_den * np.pi))


def lum_from_surf_den_R(r, s_den):

    return s_den * np.pi * r**2


df = pd.read_csv("HighzSizes/All.csv")

papers = df["Paper"].values
mags = df["Magnitude"].values
r_es_arcs = df["R_e"].values
r_es_type = df["R_e (Unit)"].values
mag_type = df["Magnitude Type"].values
zs = df["Redshift"].values

# Define pixel resolutions
wfc3 = 0.13
nircam_short = 0.031
nircam_long = 0.063

# Convert to physical kiloparsecs
r_es = np.zeros(len(papers))
for (ind, r), z in zip(enumerate(r_es_arcs), zs):
    if r_es_type[ind] == "kpc":
        r_es[ind] = r
    else:
        r_es[ind] = r / cosmo.arcsec_per_kpc_proper(z).value
    if mags[ind] < 0:
        mags[ind] = M_to_m(mags[ind], cosmo, z)

labels = {"G11": "Grazian+2011",
          "G12": "Grazian+2012",
          "C16": "Calvi+2016",
          "K18": "Kawamata+2018",
          "M18": "FIRE-2 intrinsic",
          "MO18": "Morishita+2018",
          "B19": "Bridge+2019",
          "O16": "Oesch+2016",
          "S18": "Salmon+2018",
          "H20": "Holwerda+2020"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "X", "F20": ".", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "<", "H20": "*"}
colors = {"G11": "darkred", "G12": "darkred", "C16": "darkred",
          "K18": "darkred", "M18": "green", "F20": ".",
          "MO18": "darkred", "B19": "darkred", "O16": "darkred",
          "S18": "darkred", "H20": "darkred"}

# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

if sys.argv[3] == "All":
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
else:
    snaps = sys.argv[3]

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

csoft = 0.001802390 / (0.6777) * 1e3

masslim = 10 ** float(sys.argv[4])

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
weight_dict = {}

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

    hdf = h5py.File("data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
    type_group = hdf[Type]
    orientation_group = type_group[orientation]

    hlr_dict.setdefault(snap, {})
    hlr_app_dict.setdefault(snap, {})
    hlr_pix_dict.setdefault(snap, {})
    lumin_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hlr_app_dict[snap].setdefault(f, [])
        hlr_pix_dict[snap].setdefault(f, [])
        lumin_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = orientation_group[f]["Mass"][...]
        okinds = masses > masslim

        hlr_dict[snap][f].extend(orientation_group[f]["HLR_0.5"][...][okinds])
        hlr_app_dict[snap][f].extend(
            orientation_group[f]["HLR_Aperture_0.5"][...][okinds])
        hlr_pix_dict[snap][f].extend(
            orientation_group[f]["HLR_Pixel_0.5"][...][okinds])
        lumin_dict[snap][f].extend(
            orientation_group[f]["Luminosity"][...][okinds])
        mass_dict[snap][f].extend(masses[okinds])
        weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))

    hdf.close()

for f in filters:

    fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                             np.log10(M_to_lum(-18)),
                             1000)

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)

    if len(snaps) == 9:

        axlims_x = []
        axlims_y = []
        legend_elements = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]
            try:
                cbar = ax.hexbin(lumins, hlrs,
                                 C=w,
                                 reduce_C_function=np.sum, gridsize=50,
                                 mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)
                if z == 6:
                    legend_elements.append(
                        Line2D([0], [0], color=colors["K18"], linestyle="-",
                               label=labels["K18"]))

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        ax1.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

        fig.savefig('plots/' + str(
            z) + '/HalfLightRadius_' + f + '_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.1f.png' % np.log10(masslim), bbox_inches='tight')

        plt.close(fig)

        for snap in snaps:

            legend_elements = []

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax1 = ax.twinx()
            ax1.grid(False)
            try:
                sden_lumins = np.logspace(27, 29.8)
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='Greys')
                ax1.hexbin(lumins, hlrs * cosmo.arcsec_per_kpc_proper(z).value,
                           gridsize=50, mincnt=1, C=w,
                           reduce_C_function=np.sum, xscale='log',
                           yscale='log', norm=LogNorm(), linewidths=0.2,
                           cmap='Greys', alpha=0)
                # med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                # ax.plot(lumin_bin_cents, med, color="r")
                # legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
                for sden in [10.**26, 10.**27, 10.**28, 10.**29]:
                    ax.plot(sden_lumins, r_from_surf_den(sden_lumins, sden), color="grey", linestyle="--", alpha=0.8)
                    ax.text(10**29.85, r_from_surf_den(10**29.85, sden),
                            "%.1f" % np.log10(((m_to_flux(M_to_m(lum_to_M(sden), cosmo, z)) * u.nJy *u.kpc**-2) * cosmo.kpc_proper_per_arcmin(z)**2).to(u.nJy * u.sr**-1).value),
                            verticalalignment="center",
                            horizontalalignment='left', fontsize=9,
                            color="k")
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color=colors["K18"], alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            ax.set_xlim(10**26.9, 10**30.1)

            fig.savefig(
                'plots/' + str(z) + '/HalfLightRadius_' + f + '_' + str(
                    z) + '_'
                + orientation + '_' + Type + "_" + extinction + "_"
                + '%.1f.png' % np.log10(masslim),
                bbox_inches='tight')

            plt.close(fig)

            legend_elements = []

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lum_to_M(lumins), hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lum_to_M(lumins), hlrs, weights=w, bins=M_bins, quantiles=[0.5, ])
                ax.plot(M_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_M, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(lum_to_M(fit_lumins), fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(lum_to_M(fit_lumins), up, low,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_{UV}$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            fig.savefig(
                'plots/' + str(z) + '/HalfLightRadius_AbMag_' + f + '_' + str(
                    z) + '_'
                + orientation + '_' + Type + "_" + extinction + "_"
                + '%.1f.png' % np.log10(masslim),
                bbox_inches='tight')

            plt.close(fig)

        axlims_x = []
        axlims_y = []
        legend_elements = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds] * 1000
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
            legend_elements.append(Line2D([0], [0], color=colors["K18"],
                                          label=labels["K18"]))

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        ax1.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

        fig.savefig('plots/' + str(z) + '/HalfLightRadiusAperture_'
                    + f + '_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.1f.png' % np.log10(masslim), bbox_inches='tight')

        plt.close(fig)

        for snap in snaps:

            legend_elements = []

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            fig.savefig('plots/' + str(z) + '/HalfLightRadiusAperture_'
                        + f + '_' + str(z) + '_' + orientation
                        + '_' + Type + "_" + extinction + "_"
                        + '%.1f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

            legend_elements = []

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lum_to_M(lumins), hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lum_to_M(lumins), hlrs, weights=w, bins=M_bins, quantiles=[0.5, ])
                ax.plot(M_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_M, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(lum_to_M(fit_lumins), fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(lum_to_M(fit_lumins), up, low,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_{UV}$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            fig.savefig('plots/' + str(
                z) + '/HalfLightRadiusAperture_AbMag_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.1f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

        axlims_x = []
        axlims_y = []
        legend_elements = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_pix_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds] * 1000
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
            legend_elements.append(
                Line2D([0], [0], color=colors["K18"], linestyle="-",
                       label=labels["K18"]))

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        ax1.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

        fig.savefig('plots/' + str(z) + '/HalfLightRadiusPixel_'
                    + f + '_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.1f.png' % np.log10(masslim), bbox_inches='tight')

        plt.close(fig)

        for snap in snaps:

            legend_elements = []

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_pix_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)
                plt_lumins = M_to_lum(plt_M)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_lumins, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            fig.savefig('plots/' + str(z) + '/HalfLightRadiusPixel_'
                        + f + '_' + str(z) + '_' + orientation
                        + '_' + Type + "_" + extinction + "_"
                        + '%.1f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

            legend_elements = []

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lum_to_M(lumins), hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lum_to_M(lumins), hlrs, weights=w, bins=M_bins, quantiles=[0.5, ])
                ax.plot(M_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                continue

            for p in labels.keys():
                okinds = papers == p
                plt_m = mags[okinds]
                plt_r_es = r_es[okinds]
                plt_zs = zs[okinds]

                okinds = np.logical_and(plt_zs >= (z - 0.5),
                                        np.logical_and(plt_zs < (z + 0.5),
                                                       np.logical_and(
                                                           plt_r_es > 0.08,
                                                           plt_m <= M_to_m(-16,
                                                                           cosmo,
                                                                           z))))
                plt_m = plt_m[okinds]
                plt_r_es = plt_r_es[okinds]

                if plt_m.size == 0:
                    continue
                plt_M = m_to_M(plt_m, cosmo, z)

                legend_elements.append(
                    Line2D([0], [0], marker=markers[p], color='w',
                           label=labels[p], markerfacecolor=colors[p],
                           markersize=8, alpha=0.7))

                ax.scatter(plt_M, plt_r_es,
                           marker=markers[p], label=labels[p], s=15,
                           color=colors[p], alpha=0.7)

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(lum_to_M(fit_lumins), fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                legend_elements.append(
                    Line2D([0], [0], color=colors["K18"], linestyle="-",
                           label=labels["K18"]))
                # ax.fill_between(lum_to_M(fit_lumins), up, low,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_{UV}$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(handles=legend_elements, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

            fig.savefig('plots/' + str(
                z) + '/HalfLightRadiusPixel_AbMag_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.1f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

    else:

        legend_elements = []

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])
            mass = np.array(mass_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]
            mass = mass[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(loc="bottom right")

            fig.savefig(
                'plots/' + str(z) + '/HalfLightRadius_' + f + '_' + str(
                    z) + '_'
                + orientation + '_' + Type + "_" + extinction + "_"
                + '%.1f.png' % np.log10(masslim),
                bbox_inches='tight')

            plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            try:
                cbar = ax.hexbin(mass, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 yscale='log', xscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_\star/M_\odot$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            fig.savefig(
                'plots/' + str(z) + '/HalfLightRadius_Mass_' + f + '_' + str(
                    z) + '_'
                + orientation + '_' + Type + "_" + extinction + "_"
                + '%.2f.png' % np.log10(masslim),
                bbox_inches='tight')

            plt.close(fig)

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            w = np.array(weight_dict[snap][f])[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 C=w,
                                 reduce_C_function=np.sum,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                med = util.binned_weighted_quantile(lumins, hlrs, weights=w, bins=lumin_bins, quantiles=[0.5, ])
                ax.plot(lumin_bin_cents, med, color="r")
                legend_elements.append(Line2D([0], [0], color='r', label="Weighted Median"))
            except ValueError as e:
                print(e)
                continue

            if int(z) in [6, 7, 8, 9]:

                if z == 7 or z == 6:
                    low_lim = -16
                elif z == 8:
                    low_lim = -16.8
                else:
                    low_lim = -15.4
                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(low_lim)),
                                         1000)

                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            fig.savefig('plots/' + str(z) + '/HalfLightRadiusAperture_'
                        + f + '_' + str(z) + '_' + orientation
                        + '_' + Type + "_" + extinction + "_"
                        + '%.1f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)
