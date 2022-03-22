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


# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
bt_params = {'beta': {7: 0.24, 8: 0.17, 9: 0.16, 10: 0.12, 11: 0.11},
             'r_0': {7: 0.75, 8: 0.67, 9: 0.60, 10: 0.57, 11: 0.52}}
mer_params = {'beta': {5: 0.32, 6: 0.34, 7: 0.32, 8: 0.33, 9: 0.32, 10: 0.36},
              'r_0': {5: 1.53, 6: 1.14, 7: 0.85, 8: 0.66, 9: 0.49, 10: 0.43}}
bt_fit_uplims = {7: 30, 8: 29.8, 9: 29.72, 10: 29.61, 11: 29.6}
# https://iopscience.iop.org/article/10.1088/0004-637X/765/1/68
huang_params = {'beta': {5: 0.25, }, 'r_0': {5: 1.19, }}
# https://iopscience.iop.org/article/10.1088/0004-637X/808/1/6
holw_params = {'beta': {7: 0.24, 9: 0.12, 10: 0.12},
               'r_0': {7: 0.86, 9: 0.57, 10: 0.57}}
# https://arxiv.org/pdf/2201.08858.pdf
yang_params = {'beta': {6: 0.48, 7: 0.48},
               'r_0': {6: 0.69, 7: 0.69}}
# https://arxiv.org/pdf/2112.02948.pdf
bouw_params = {'beta': {6: 0.4, 7: 0.4, 8: 0.4},
               'r_0': {6: 10**2.76 / 1000,
                       7: 10**2.76 / 1000,
                       8: 10**2.76 / 1000}}

# Lstar = M_to_lum(-21)
Lstar = M_to_lum(-21)

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

cmap = mpl.cm.get_cmap("plasma")
norm = plt.Normalize(vmin=0, vmax=1)

labels = {"H07": "Hathi+2008",
          "G11": "Grazian+2011",
          "G12": "Grazian+2012",
          "Hu13": "Huang+2013",
          "H15": "Holwerda+2015",
          "C16": "Calvi+2016",
          "K18": "Kawamata+2018",
          "MO18": "Morishita+2018",
          "B19": "Bridge+2019",
          "B21": "Bouwens+2021",
          "Y22": "Yang+2022"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "X", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "<", "H20": "*",
           "H07": "P", }
colors = {}
for key, col in zip(labels.keys(), np.linspace(0, 1, len(labels.keys()))):
    colors[key] = cmap(norm(col))

csoft = 0.001802390 / (0.6777) * 1e3

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)


def fit_size_lumin_grid(data, intr_data, snaps, filters, orientation, Type,
                        extinction, mtype, sample, xlims, ylims, extent):
    for f in filters:

        print("Plotting for:")
        print("Orientation =", orientation)
        print("Type =", Type)
        print("Filter =", f)

        fig = plt.figure(figsize=(2.25 * len(snaps), 2.25))
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
                intr_hlrs = np.array(intr_data[snap][f]["HLR_0.5"])
                intr_lumins = np.array(intr_data[snap][f]["Luminosity"])
            elif mtype == "app":
                hlrs = np.array(data[snap][f]["HLR_Aperture_0.5"])
                intr_hlrs = np.array(intr_data[snap][f]["HLR_Aperture_0.5"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])
            else:
                hlrs = np.array(data[snap][f]["HLR_Pixel_0.5"])
                intr_hlrs = np.array(intr_data[snap][f]["HLR_Pixel_0.5"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])
            w = np.array(data[snap][f]["Weight"])
            intr_w = np.array(intr_data[snap][f]["Weight"])
            mass = np.array(data[snap][f]["Mass"])

            compact_ncom = data[snap][f]["Compact_Population_NotComplete"]
            diffuse_ncom = data[snap][f]["Diffuse_Population_NotComplete"]
            compact_com = data[snap][f]["Compact_Population_Complete"]
            diffuse_com = data[snap][f]["Diffuse_Population_Complete"]
            intr_compact_com = intr_data[snap][f][
                "Compact_Population_Complete"]
            intr_diffuse_com = intr_data[snap][f][
                "Diffuse_Population_Complete"]
            if sample == "Complete":
                complete = np.logical_or(compact_com, diffuse_com)
                intr_complete = np.logical_or(intr_compact_com,
                                              intr_diffuse_com)
            else:
                complete = data[snap][f]["okinds"]
                intr_complete = intr_data[snap][f]["okinds"]

            try:

                popt, pcov = curve_fit(r_fit, lumins[complete],
                                       hlrs[complete],
                                       p0=(kawa_params['r_0'][7],
                                           kawa_params['beta'][7]),
                                       sigma=w[complete])
            except ValueError as e:
                print(e)

            try:

                intr_popt, intr_pcov = curve_fit(r_fit,
                                                 intr_lumins[intr_complete],
                                                 intr_hlrs[intr_complete],
                                                 p0=(kawa_params['r_0'][7],
                                                     kawa_params['beta'][7]),
                                                 sigma=intr_w[intr_complete])
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

                print("--------------", "Compact", mtype, f, snap,
                      "--------------")
                print("R_0=%.3f+/-%.3f" % (popt1[0],
                                           np.sqrt(pcov1[0, 0])))
                print("beta=%.3f+/-%.3f" % (popt1[1],
                                            np.sqrt(pcov1[1, 1])))
                print("N=", lumins[compact_com].size)
                print(
                    "------------------------------------------"
                    "----------------")

                print("--------------", "Total", mtype, f, snap,
                      "--------------")
                print("R_0=%.3f+/-%.3f" % (popt[0],
                                           np.sqrt(pcov[0, 0])))
                print("beta=%.3f+/-%.3f" % (popt[1],
                                           np.sqrt(pcov[1, 1])))
                print("N=", lumins[complete].size)
                print(
                    "-------------------------------------"
                    "---------------------")

                fit_lumins = np.logspace(np.log10(np.min(lumins[complete])),
                                         np.log10(np.max(lumins[complete])),
                                         1000)

                print(np.log10(fit_lumins[0]), np.log10(fit_lumins[-1]))

                fit = r_fit(fit_lumins, popt[0], popt[1])

                axes[i].plot(fit_lumins, fit,
                             linestyle='-', color="r",
                             zorder=3)

                # fit = r_fit(fit_lumins, popt1[0], popt1[1])
                #
                # axes[i].plot(fit_lumins, fit,
                #              linestyle='-.', color="r",
                #              zorder=3)

                # fit = r_fit(fit_lumins, popt1[0], popt1[1])
                #
                # axes[i].plot(fit_lumins, fit,
                #              linestyle='-', color="m",
                #              zorder=3,
                #              linewidth=2)

            except ValueError as e:
                print(e)

            if int(z) in kawa_params["beta"]:

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
                             linestyle='dashed', color=colors["K18"],
                             zorder=2,
                             label="Kawamata+18")

            if int(z) in huang_params["beta"]:

                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(-16)),
                                         1000)

                fit = kawa_fit(fit_lumins, huang_params['r_0'][int(z)],
                               huang_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dashed', color=colors["Hu13"],
                             zorder=2,
                             label=labels["Hu13"])

            if int(z) in holw_params["beta"]:

                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(-16)),
                                         1000)

                fit = kawa_fit(fit_lumins, holw_params['r_0'][int(z)],
                               holw_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dashed', color=colors["H15"],
                             zorder=2,
                             label=labels["H15"])

            if int(z) in yang_params["beta"]:

                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(-16)),
                                         1000)

                fit = kawa_fit(fit_lumins, yang_params['r_0'][int(z)],
                               yang_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dashed', color=colors["Y22"],
                             zorder=2,
                             label=labels["Y22"])

            if int(z) in bouw_params["beta"]:

                fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                                         np.log10(M_to_lum(-16)),
                                         1000)

                fit = kawa_fit(fit_lumins, bouw_params['r_0'][int(z)],
                               bouw_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dashed', color=colors["B21"],
                             zorder=2,
                             label=labels["B21"])

            if int(z) in [7, 8, 9, 10, 11]:
                fit_lumins = np.logspace(28.5,
                                         bt_fit_uplims[int(z)],
                                         1000)

                fit = r_fit(fit_lumins, bt_params['r_0'][int(z)],
                            bt_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dotted', color="b",
                             zorder=2,
                             label="Marshall+21")

            if int(z) in [5, 6, 7, 8, 9, 10]:
                fit_lumins = np.logspace(np.log10(M_to_lum(-22)),
                                         np.log10(M_to_lum(-14)),
                                         1000)

                fit = r_fit(fit_lumins, mer_params['r_0'][int(z)],
                            mer_params['beta'][int(z)])
                axes[i].plot(fit_lumins, fit,
                             linestyle='dotted', color="g",
                             zorder=2,
                             label="Marshall+19")

            axes[i].text(0.95, 0.05, f'$z={z}$',
                         bbox=dict(boxstyle="round,pad=0.3", fc='w',
                                   ec="k", lw=1, alpha=0.8),
                         transform=axes[i].transAxes,
                         horizontalalignment='right',
                         fontsize=8)

            axes[i].tick_params(axis='both', which='minor',
                                bottom=True, left=True)

            # Label axes
            axes[i].set_xlabel(r"$L_{" + f.split(".")[-1]
                               + "}/$ [erg $/$ s $/$ Hz]")

            axes[i].tick_params(axis='x', which='both', bottom=True)

            axes[i].set_xlim(10 ** extent[2], 10 ** extent[3])

        for i in range(len(axes)):
            axes[i].set_ylim(10 ** extent[0], 10 ** extent[1])

        axes[0].set_ylabel('$R/ [pkpc]$')
        axes[0].tick_params(axis='y', which='both', left=True)

        uni_legend_elements = []

        uni_legend_elements.append(
            Line2D([0], [0], color="r", linestyle="-",
                   label="FLARES"))
        # uni_legend_elements.append(
        #     Line2D([0], [0], color="r", linestyle="-.",
        #            label="FLARES (Compact)"))

        uni_legend_elements.append(
            Line2D([0], [0], color=colors["Hu13"], linestyle="--",
                   label=labels["Hu13"]))
        uni_legend_elements.append(
            Line2D([0], [0], color=colors["H15"], linestyle="--",
                   label=labels["H15"]))
        uni_legend_elements.append(
            Line2D([0], [0], color=colors["K18"], linestyle="--",
                   label=labels["K18"]))
        uni_legend_elements.append(
            Line2D([0], [0], color=colors["B21"], linestyle="--",
                   label=labels["B21"]))
        uni_legend_elements.append(
            Line2D([0], [0], color=colors["Y22"], linestyle="--",
                   label=labels["Y22"]))

        uni_legend_elements.append(
            Line2D([0], [0], color="b", linestyle="dotted",
                   label="Marshall+2021"))
        uni_legend_elements.append(
            Line2D([0], [0], color="g", linestyle="dotted",
                   label="Marshall+2019"))
        included = []
        for l in legend_elements:
            if (l.get_label(), l.get_marker()) not in included:
                print((l.get_label(), l.get_marker()))
                uni_legend_elements.append(l)
                included.append((l.get_label(), l.get_marker()))

        axes[2].legend(handles=uni_legend_elements, loc='upper center',
                       bbox_to_anchor=(0.5, -0.15), fancybox=True,
                       ncol=2)

        fig.savefig(
            'plots/FitHalfLightRadius_' + mtype + "_" + f + '_'
            + orientation + '_' + Type + "_" + extinction + "_"
            + sample + ".pdf",
            bbox_inches='tight', dpi=300)

        plt.close(fig)
