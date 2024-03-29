#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import pandas as pd
from flare import plt as flareplt
plt.rcParams['axes.grid'] = True


# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
bt_params = {'beta': {7: 0.24, 8: 0.17, 9: 0.16, 10: 0.12, 11: 0.11},
             'r_0': {7: 0.75, 8: 0.67, 9: 0.60, 10: 0.57, 11: 0.52}}
mer_params = {'beta': {5: 0.25, 6: 0.23, 7: 0.25, 8: 0.28, 9: 0.30, 10: 0.36},
             'r_0': {5: 1.17, 6: 0.80, 7: 0.61, 8: 0.53, 9: 0.42, 10: 0.45}}
bt_fit_uplims = {7: 30, 8: 29.8, 9: 29.72, 10: 29.61, 11: 29.6}

# Lstar = M_to_lum(-21)
Lstar = 10**28.51

r_fit = lambda l, r0, b: r0 * (l / Lstar) ** b
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
    return s_den * np.pi * r ** 2


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

cmap = mpl.cm.get_cmap("winter")
norm = plt.Normalize(vmin=0, vmax=1)

labels = {"G11": "Grazian+2011",
          "G12": "Grazian+2012",
          "C16": "Calvi+2016",
          "K18": "Kawamata+2018",
          "MO18": "Morishita+2018",
          "B19": "Bridge+2019",
          #"O16": "Oesch+2016",
          # "S18": "Salmon+2018",
          # "H20": "Holwerda+2020",
          "H07": "Hathi+2007"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "X", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "<", "H20": "*",
           "H07": "P"}
colors = {}
for key, col in zip(markers.keys(), np.linspace(0, 1, len(markers.keys()))):
    colors[key] = cmap(norm(col))

csoft = 0.001802390 / (0.6777) * 1e3

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)


def fit_size_lumin_grid_nosmooth(data, snaps, filters, orientation, Type,
                                 extinction, mtype, sample, xlims, ylims,
                                 weight_norm):

    extent = (np.log10(xlims[0]), np.log10(xlims[1]),
              np.log10(ylims[0]), np.log10(ylims[1]))

    for f in filters:

        print("Plotting for:")
        print("Orientation =", orientation)
        print("Type =", Type)
        print("Filter =", f)

        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, len(snaps))
        gs.update(wspace=0.0, hspace=0.0)
        axes = []
        i = 0
        while i < len(snaps):
            axes.append(fig.add_subplot(gs[0, i]))
            axes[-1].loglog()
            if i > 0:
                axes[-1].tick_params(axis='y', left=False, right=False,
                                     labelleft=False, labelright=False)
            i += 1

        legend_elements = []

        for i, snap in enumerate(snaps):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            if mtype == "part":
                hlrs = np.array(data[snap][f]["HLR_0.5"])
                lumins = np.array(data[snap][f]["Luminosity"])
                intr_lumins = np.array(data[snap][f]["Luminosity"])
            elif mtype == "app":
                hlrs = np.array(data[snap][f]["HLR_Aperture_0.5"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(data[snap][f]["Image_Luminosity"])
            else:
                hlrs = np.array(data[snap][f]["HLR_Pixel_0.5_No_Smooth"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(data[snap][f]["Image_Luminosity_No_Smooth"])
            w = np.array(data[snap][f]["Weight"])
            mass = np.array(data[snap][f]["Mass"])

            compact_ncom = data[snap][f]["Compact_Population_NotComplete"]
            diffuse_ncom = data[snap][f]["Diffuse_Population_NotComplete"]
            compact_com = data[snap][f]["Compact_Population_Complete"]
            diffuse_com = data[snap][f]["Diffuse_Population_Complete"]
            if sample == "Complete":
                complete = np.logical_or(compact_com, diffuse_com)
            else:
                complete = data[snap][f]["okinds"]

            try:
                cbar = axes[i].hexbin(lumins[diffuse_ncom], hlrs[diffuse_ncom],
                                      gridsize=50,
                                      mincnt=1, C=w[diffuse_ncom],
                                      reduce_C_function=np.sum,
                                      xscale='log', yscale='log',
                                      norm=weight_norm, linewidths=0.2,
                                      cmap='Greys', alpha=0.2, extent=extent)
            except ValueError as e:
                print(e, "Diffuse incomplete", snap, f)
                print(lumins[diffuse_ncom][lumins[diffuse_ncom] <= 0],
                      lumins[diffuse_ncom][lumins[diffuse_ncom] <= 0].size)
            try:
                axes[i].hexbin(lumins[compact_ncom], hlrs[compact_ncom],
                               gridsize=50,
                               mincnt=1,
                               C=w[compact_ncom],
                               reduce_C_function=np.sum,
                               xscale='log', yscale='log',
                               norm=weight_norm, linewidths=0.2,
                               cmap='plasma', alpha=0.2, extent=extent)
            except ValueError as e:
                print(e, "Compact incomplete", snap, f)
                print(lumins[compact_ncom][lumins[compact_ncom] <= 0],
                      lumins[compact_ncom][lumins[compact_ncom] <= 0].size)
            try:
                cbar = axes[i].hexbin(lumins[diffuse_com], hlrs[diffuse_com],
                                      gridsize=50,
                                      mincnt=1, C=w[diffuse_com],
                                      reduce_C_function=np.sum,
                                      xscale='log', yscale='log',
                                      norm=weight_norm, linewidths=0.2,
                                      cmap='Greys', extent=extent)
            except ValueError as e:
                print(e, "Diffuse complete", snap, f)
            try:
                axes[i].hexbin(lumins[compact_com],
                               hlrs[compact_com], gridsize=50,
                               mincnt=1,
                               C=w[compact_com],
                               reduce_C_function=np.sum,
                               xscale='log', yscale='log',
                               norm=weight_norm, linewidths=0.2,
                               cmap='plasma', extent=extent)
            except ValueError as e:
                print(e, "Compact complete", snap, f)

            try:

                popt, pcov = curve_fit(r_fit, lumins[complete],
                                       hlrs[complete],
                                       p0=(kawa_params['r_0'][7],
                                           kawa_params['beta'][7]),
                                       sigma=w[complete])
            except ValueError as e:
                print(e)

            try:
                popt1, pcov1 = curve_fit(r_fit, lumins[compact_com],
                                         hlrs[compact_com],
                                         p0=(kawa_params['r_0'][7],
                                             kawa_params['beta'][7]),
                                         sigma=w[compact_com])
            except ValueError as e:
                print(e)

            try:
                popt2, pcov2 = curve_fit(r_fit, lumins,
                                         hlrs,
                                         p0=(kawa_params['r_0'][7],
                                             kawa_params['beta'][7]),
                                         sigma=w)
            except ValueError as e:
                print(e)

            try:
                print("--------------", "Total", "No smooth", Type, mtype,
                      f, snap,
                      "--------------")
                print("R_0=", popt[0], "+/-", np.sqrt(pcov[0, 0]))
                print("beta=", popt[1], "+/-", np.sqrt(pcov[1, 1]))

                # print("--------------", "Total", "Compact", mtype, f,
                #       "--------------")
                # print("R_0=", popt1[0], "+/-", np.sqrt(pcov1[0, 0]))
                # print("beta=", popt1[1], "+/-", np.sqrt(pcov1[1, 1]))
                # print(pcov1)
                print("--------------", "Kawamata", "All", mtype, f,
                      "--------------")
                if int(z) in [6, 7, 8, 9]:
                    print("R_0=", kawa_params['r_0'][int(z)])
                    print("beta=", kawa_params['beta'][int(z)])
                else:
                    print("R_0= ---")
                    print("beta= ---")
                print(
                    "----------------------------------------------------------")

                fit_lumins = np.logspace(np.log10(np.min(lumins[complete])),
                                         np.log10(np.max(lumins[complete])),
                                         1000)

                fit = r_fit(fit_lumins, popt[0], popt[1])

                axes[i].plot(fit_lumins, fit,
                             linestyle='-', color="r",
                             zorder=3)

                # fit = r_fit(fit_lumins, popt1[0], popt1[1])
                #
                # axes[i].plot(fit_lumins, fit,
                #              linestyle='-', color="m",
                #              zorder=3,
                #              linewidth=2)

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
                axes[i].plot(fit_lumins, fit,
                             linestyle='dashed', color="g",
                             zorder=2,
                             label="Kawamata+18")

            # if int(z) in [7, 8, 9, 10, 11]:
            #
            #     fit_lumins = np.logspace(28.5,
            #                              bt_fit_uplims[int(z)],
            #                              1000)
            #
            #     fit = r_fit(fit_lumins, bt_params['r_0'][int(z)],
            #                    bt_params['beta'][int(z)])
            #     axes[i].plot(fit_lumins, fit,
            #                  linestyle='dashed', color="b",
            #                  zorder=2,
            #                  label="Marshall+21")
            #
            # if int(z) in [5, 6, 7, 8, 9, 10]:
            #
            #     fit_lumins = np.logspace(np.log10(M_to_lum(-22)),
            #                              np.log10(M_to_lum(-14)),
            #                              1000)
            #
            #     fit = r_fit(fit_lumins, mer_params['r_0'][int(z)],
            #                    mer_params['beta'][int(z)])
            #     axes[i].plot(fit_lumins, fit,
            #                  linestyle='dashed', color="m",
            #                  zorder=2,
            #                  label="Liu+17")

            axes[i].text(0.95, 0.05, f'$z={z}$',
                         bbox=dict(boxstyle="round,pad=0.3", fc='w',
                                   ec="k", lw=1, alpha=0.8),
                         transform=axes[i].transAxes,
                         horizontalalignment='right',
                         fontsize=8)

            axes[i].tick_params(axis='both', which='both',
                                bottom=True, left=True)

            # Label axes
            axes[i].set_xlabel(r"$L_{" + f.split(".")[-1]
                               + "}/$ [erg $/$ s $/$ Hz]")

            axes[i].set_xlim(xlims[0], xlims[1])

        for i in range(len(axes)):
            axes[i].set_ylim(ylims[0], ylims[1])

        axes[0].set_ylabel('$R_{1/2}/ [pkpc]$')
        axes[0].tick_params(axis='y', which='both', left=True)

        uni_legend_elements = []
        # uni_legend_elements.append(
        #     Line2D([0], [0], color="m", linestyle="dotted",
        #            label="FLARES (All)"))
        uni_legend_elements.append(
            Line2D([0], [0], color="r", linestyle="-",
                   label="FLARES"))
        # uni_legend_elements.append(
        #     Line2D([0], [0], color="m", linestyle="--",
        #            label="FLARES ($M_{\star}/M_\odot<10^{9}$)"))
        uni_legend_elements.append(
            Line2D([0], [0], color="g", linestyle="--",
                   label=labels["K18"]))
        # uni_legend_elements.append(
        #     Line2D([0], [0], color="b", linestyle="--",
        #            label="Marshall+2021"))
        # uni_legend_elements.append(
        #     Line2D([0], [0], color="m", linestyle="--",
        #            label="Liu+2017"))
        included = []
        for l in legend_elements:
            if (l.get_label(), l.get_marker()) not in included:
                print((l.get_label(), l.get_marker()))
                uni_legend_elements.append(l)
                included.append((l.get_label(), l.get_marker()))

        axes[2].legend(handles=uni_legend_elements, loc='upper center',
                       bbox_to_anchor=(0.5, -0.15), fancybox=True,
                       ncol=len(uni_legend_elements))

        fig.savefig(
            'plots/FitHalfLightRadius_NoSmooth_' + mtype + "_" + f + '_'
            + orientation + '_' + Type + "_" + extinction + "_"
            + sample + ".pdf",
            bbox_inches='tight', dpi=300)

        plt.close(fig)
