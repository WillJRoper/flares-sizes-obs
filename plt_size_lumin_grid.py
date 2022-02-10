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
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import pandas as pd
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

# Lstar = M_to_lum(-21)
Lstar = 10**28.51

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
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

    axes[i].plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
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
colors_in_order = []
for key, col in zip(labels.keys(), np.linspace(0, 1, len(labels.keys()))):
    colors[key] = cmap(norm(col))
    colors_in_order.append(key)

csoft = 0.001802390 / (0.6777) * 1e3

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)


def size_lumin_grid(data, snaps, filters, orientation, Type, extinction,
                    mtype, weight_norm, xlims, ylims, extent):

    for f in filters:

        print("Plotting for:")
        print("Orientation =", orientation)
        print("Type =", Type)
        print("Filter =", f)

        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(2, len(snaps))
        gs.update(wspace=0.0, hspace=0.0)
        axes_diff = []
        axes_com = []
        i = 0
        while i < len(snaps):
            axes_com.append(fig.add_subplot(gs[0, i]))
            axes_diff.append(fig.add_subplot(gs[1, i]))
            if i > 0:
                axes_com[-1].tick_params(axis='both', left=False, right=False,
                                         top=False, bottom=False,
                                         labelleft=False, labelright=False,
                                         labeltop=False, labelbottom=False)
                axes_diff[-1].tick_params(axis='y', left=False, right=False,
                                         labelleft=False, labelright=False)
            else:
                axes_com[-1].tick_params(axis='x',
                                         top=False, bottom=False,
                                         labeltop=False, labelbottom=False)
            i += 1

        legend_elements = []

        for i, snap in enumerate(snaps):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            if z <= 2.8:
                csoft = 0.000474390 / 0.6777 * 1e3
            else:
                csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

            if mtype == "part":
                hlrs = np.array(data[snap][f]["HLR_0.5"])
                lumins = np.array(data[snap][f]["Luminosity"])
                intr_lumins = np.array(data[snap][f]["Luminosity"])
            elif mtype == "app":
                hlrs = np.array(data[snap][f]["HLR_Aperture_0.5"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(data[snap][f]["Image_Luminosity"])
            else:
                hlrs = np.array(data[snap][f]["HLR_Pixel_0.5"])
                lumins = np.array(data[snap][f]["Image_Luminosity"])
                intr_lumins = np.array(data[snap][f]["Image_Luminosity"])
            w = np.array(data[snap][f]["Weight"])
            mass = np.array(data[snap][f]["Mass"])

            compact_ncom = data[snap][f]["Compact_Population_NotComplete"]
            diffuse_ncom = data[snap][f]["Diffuse_Population_NotComplete"]
            compact_com = data[snap][f]["Compact_Population_Complete"]
            diffuse_com = data[snap][f]["Diffuse_Population_Complete"]

            try:
                cbar = axes_diff[i].hexbin(lumins[diffuse_com], hlrs[diffuse_com],
                                      gridsize=50,
                                      mincnt=np.min(w) - (0.1 * np.min(w)),
                                      C=w[diffuse_com],
                                      reduce_C_function=np.sum,
                                      xscale='log', yscale='log',
                                      norm=weight_norm, linewidths=0.2,
                                      cmap='Greys', extent=[extent[2],
                                                            extent[3],
                                                            extent[0],
                                                            extent[1]])
                print(np.log10(np.min(lumins[diffuse_com])),
                      np.log10(np.max(lumins[diffuse_com])),
                      np.log10(np.min(lumins[compact_com])),
                      np.log10(np.max(lumins[compact_com])))
            except ValueError as e:
                print(e, "Diffuse complete", snap, f)
            try:
                axes_com[i].hexbin(lumins[compact_com],
                               hlrs[compact_com], gridsize=50,
                               mincnt=np.min(w) - (0.1 * np.min(w)),
                               C=w[compact_com],
                               reduce_C_function=np.sum,
                               xscale='log', yscale='log',
                               norm=weight_norm, linewidths=0.2,
                               cmap='viridis', extent=[extent[2], extent[3],
                                                       extent[0], extent[1]])
            except ValueError as e:
                print(e, "Compact complete", snap, f)

            if Type != "Intrinsic":

                for p in colors_in_order:
                    okinds = papers == p
                    plt_m = mags[okinds]
                    plt_r_es = r_es[okinds]
                    plt_zs = zs[okinds]

                    okinds = np.logical_and(plt_zs >= (z - 0.5),
                                            np.logical_and(plt_zs < (z + 0.5),
                                                           np.logical_and(
                                                               plt_r_es > 0.08,
                                                               plt_m <= M_to_m(
                                                                   -16,
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
                               markersize=8, alpha=0.9))

                    axes_com[i].scatter(plt_lumins, plt_r_es,
                                        marker=markers[p], label=labels[p], s=20,
                                        color=colors[p], alpha=0.9)
                    axes_diff[i].scatter(plt_lumins, plt_r_es,
                                        marker=markers[p], label=labels[p],
                                        s=20,
                                        color=colors[p], alpha=0.9)

            axes_diff[i].text(0.95, 0.05, f'$z={z}$',
                         bbox=dict(boxstyle="round,pad=0.3", fc='w',
                                   ec="k", lw=1, alpha=0.8),
                         transform=axes_diff[i].transAxes,
                         horizontalalignment='right',
                         fontsize=8)

            axes_diff[i].tick_params(axis='both', which='minor',
                                     bottom=True, left=True)
            axes_com[i].tick_params(axis='both', which='minor',
                                     bottom=True, left=True)

            # Label axes
            axes_diff[i].set_xlabel(r"$L_{" + f.split(".")[-1]
                               + "}/$ [erg $/$ s $/$ Hz]")

            axes_diff[i].set_xlim(10 ** extent[2], 10 ** extent[3])
            axes_com[i].set_xlim(10 ** extent[2], 10 ** extent[3])

            axes_diff[i].axhline(csoft, linestyle="--", color="k")
            axes_com[i].axhline(csoft, linestyle="--", color="k")

        for i in range(len(axes_diff)):
            axes_diff[i].set_ylim(10 ** extent[0], 10 ** extent[1])
            axes_com[i].set_ylim(10 ** extent[0], 10 ** extent[1])

        axes_diff[0].set_ylabel('$R / [pkpc]$')
        axes_com[0].set_ylabel('$R / [pkpc]$')

        uni_legend_elements = []
        uni_legend_elements.append(
            Line2D([0], [0], color="k", linestyle="none", marker="h",
                   label="FLARES"))
        included = []
        for l in legend_elements:
            if (l.get_label(), l.get_marker()) not in included:
                print((l.get_label(), l.get_marker()))
                uni_legend_elements.append(l)
                included.append((l.get_label(), l.get_marker()))

        axes_diff[2].legend(handles=uni_legend_elements, loc='upper center',
                       bbox_to_anchor=(0.5, -0.15), fancybox=True,
                       ncol=len(uni_legend_elements))

        ax2 = fig.add_axes([0.91, 0.1, 0.01, 0.8])
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap("Greys"),
                                        norm=weight_norm)
        cb1.set_label("$\sum w_{i}$")

        ax2 = fig.add_axes([0.96, 0.1, 0.01, 0.8])
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=plt.get_cmap("viridis"),
                                        norm=weight_norm)
        cb1.set_label("$\sum w_{i}$")

        fig.savefig(
            'plots/HalfLightRadius_' + mtype + "_" + f + '_'
            + orientation + '_' + Type + "_" + extinction + ".pdf",
            bbox_inches='tight', dpi=300)

        plt.close(fig)
