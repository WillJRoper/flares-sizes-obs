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
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.colors as cm
import cmasher as cmr
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

filter_path = "/cosma7/data/dp004/dc-wilk2/flare/data/filters/"

Lstar = M_to_lum(-21)

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b
r_fit = lambda l, r0, b: r0 * (l / Lstar) ** b
st_line_fit = lambda x, m, c: m * np.log10(x) + c

bt_fits = {7: {"FUV": (0.2421, 0.7544), "NUV": (0.1558, 0.6697),
               "U": (0.0689, 0.6065), "B": (0.0266, 0.5983),
               "V": (-0.0058, 0.5906), "I": (-0.0324, 0.5672),
               "Z": (-0.0367, 0.5542), "Y": (-0.0405, 0.5568),
               "J": (-0.0563, 0.5476), "H": (-0.0669, 0.5372)},
           8: {"FUV": (0.1655, 0.6713), "NUV": (0.1077, 0.6134),
               "U": (0.0406, 0.5652), "B": (0.0013, 0.5616),
               "V": (-0.0187, 0.5534), "I": (-0.0328, 0.5316),
               "Z": (-0.0367, 0.5171), "Y": (-0.0395, 0.5215),
               "J": (-0.0483, 0.5129), "H": (-0.0544, 0.5026)}
           }


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
          "O16": "Oesch+2016",
          "S18": "Salmon+2018",
          "H20": "Holwerda+2020"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "X", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "<", "H20": "*"}

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


def size_lumin_grid_allf(data, intr_data, snaps, filters, orientation,
                         Type, extinction, mtype, weight_norm, xlims, ylims,
                         sample, extent):
    trans = {}
    plt_lams = []
    cents = []
    bounds = []
    lam_max = 0
    i = 1
    for f in filters:
        l, t = np.loadtxt(filter_path + '/' + '/'.join(f.split('.')) + '.txt',
                          skiprows=1).T
        l /= 10000  # Angstrom to microns
        wid = np.max(l[t > 0]) - np.min(l[t > 0])
        trans[f] = []
        trans[f].append(np.min(l[t > 0]))
        trans[f].append(np.max(l[t > 0]) - (wid / 2))
        trans[f].append(np.max(l[t > 0]))
        plt_lams.append(np.max(l[t > 0]) - (wid / 2))
        cents.append(i)
        bounds.append(i - 0.5)
        print(f.split(".")[-1], np.min(l[t > 0]), np.max(l[t > 0]))
        if np.max(l[t > 0]) > lam_max:
            lam_max = np.max(l[t > 0])
        i += 1

    cmap = mpl.cm.get_cmap("coolwarm")

    cmaps = {filters[0]: "Blues", filters[-1]: "Reds"}

    bounds.append(i + 0.5)

    sinds = np.argsort(plt_lams)
    filters = np.array(filters)[sinds]

    norm = cm.Normalize(vmin=min(plt_lams),
                        vmax=max(plt_lams),
                        clip=True)

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)

    fig = plt.figure(figsize=(2.25 * len(snaps), 2 * 2.25))
    gs = gridspec.GridSpec(2, len(snaps) + 1, height_ratios=(5, 2),
                           width_ratios=[20, ] * len(snaps) + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    axes_ratio = []
    cax = fig.add_subplot(gs[:, -1])
    ylims_ratio = []
    i = 0
    while i < len(snaps):
        axes.append(fig.add_subplot(gs[0, i]))
        axes_ratio.append(fig.add_subplot(gs[1, i]))
        axes[-1].loglog()
        axes_ratio[-1].semilogx()
        if i > 0:
            axes[-1].tick_params(axis='y', left=False, right=False,
                                 labelleft=False, labelright=False)
            axes[-1].tick_params(axis='x', top=False, bottom=False,
                                 labeltop=False, labelbottom=False)
            axes_ratio[-1].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
        i += 1

    legend_elements = []

    for i, snap in enumerate(snaps):

        print(
            "---------------------------", snap, "---------------------------")

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        for f in filters:
            print(f)

            compact_ncom = data[snap][f]["Compact_Population_NotComplete"]
            diffuse_ncom = data[snap][f]["Diffuse_Population_NotComplete"]
            compact_com = data[snap][f]["Compact_Population_Complete"]
            diffuse_com = data[snap][f]["Diffuse_Population_Complete"]

            if sample == "Complete":
                complete = np.logical_or(compact_com, diffuse_com)
            else:
                complete = data[snap][f]["okinds"]

            if mtype == "part":
                hlrs = np.array(data[snap][f]["HLR_0.5"])[complete]
                lumins = np.array(data[snap][f]["Image_Luminosity"])[complete]
                intr_hlrs = np.array(intr_data[snap][f]["HLR_0.5"])[complete]
                intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])[
                    complete]
            elif mtype == "app":
                hlrs = np.array(data[snap][f]["HLR_Aperture_0.5"])[complete]
                lumins = np.array(data[snap][f]["Image_Luminosity"])[complete]
                intr_hlrs = np.array(intr_data[snap][f]["HLR_Aperture_0.5"])[
                    complete]
                intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])[
                    complete]
            else:
                hlrs = np.array(data[snap][f]["HLR_Pixel_0.5"])[complete]
                lumins = np.array(data[snap][f]["Image_Luminosity"])[complete]
                intr_hlrs = np.array(intr_data[snap][f]["HLR_Pixel_0.5"])[
                    complete]
                intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])[
                    complete]
            low_lum = data[snap][f]["Complete_Luminosity"]
            w = np.array(data[snap][f]["Weight"])[complete]
            mass = np.array(data[snap][f]["Mass"])[complete]

            try:
                fit_lumins = np.logspace(np.log10(low_lum),
                                         np.log10(np.max(lumins)),
                                         1000)

                popt, pcov = curve_fit(r_fit, lumins,
                                       hlrs,
                                       p0=(0.5, 0),
                                       sigma=w)

                fit = r_fit(fit_lumins, popt[0], popt[1])
                print(snap, "Total [R_0, Beta]",
                      "[%.3f +/- %.3f & %.3f +/- %.3f], N=%d"
                      % (popt[0], np.sqrt(pcov[0, 0]),
                         popt[1], np.sqrt(pcov[1, 1]), lumins.size))
                axes[i].plot(fit_lumins, fit,
                             linestyle='-', color=cmap(norm(trans[f][1])),
                             alpha=0.9, zorder=2,
                             label=f.split(".")[-1])
                if int(z) in [7, 8]:
                    if f.split(".")[-1] in bt_fits[int(z)].keys():
                        bt_fit_lumins = np.logspace(np.log10(np.min(lumins)),
                                                 np.log10(np.max(lumins)),
                                                 1000)
                        fit = r_fit(fit_lumins,
                                    bt_fits[int(z)][f.split(".")[-1]][1],
                                    bt_fits[int(z)][f.split(".")[-1]][0])
                        print("BT", bt_fits[int(z)][f.split(".")[-1]])
                        axes[i].plot(bt_fit_lumins, fit,
                                     linestyle='--',
                                     color=cmap(norm(trans[f][1])),
                                     alpha=0.6, zorder=1)

            except ValueError as e:
                print(e, f, "Total")

            try:
                popt, pcov = curve_fit(st_line_fit, lumins,
                                       hlrs / intr_hlrs,
                                       p0=(1, 1),
                                       sigma=w)

                print("Ratio", popt)
                fit = st_line_fit(fit_lumins, popt[0], popt[1])

                axes_ratio[i].plot(fit_lumins, fit,
                                   linestyle='-',
                                   color=cmap(norm(trans[f][1])),
                                   alpha=0.9, zorder=1,
                                   label=f.split(".")[-1])
            except ValueError as e:
                print(e, f, "Intrinsic")

            # if f == filters[-1]:
            #     print(f, np.log10(np.min(lumins)), np.log10(np.max(lumins)))
            #     axes[i].hexbin(lumins,
            #                    hlrs, gridsize=50,
            #                    mincnt=0.00001,
            #                    C=w,
            #                    reduce_C_function=np.sum,
            #                    xscale='log', yscale='log',
            #                    norm=weight_norm, linewidths=0.2,
            #                    cmap=cmaps[f], alpha=0.5)

        axes[i].text(0.95, 0.1, f'$z={z}$',
                     bbox=dict(boxstyle="round,pad=0.3", fc='w',
                               ec="k", lw=1, alpha=0.8),
                     transform=axes[i].transAxes,
                     horizontalalignment='right',
                     fontsize=8)

        axes[i].tick_params(axis='y', which='minor', left=True)
        axes_ratio[i].tick_params(axis='both', which='minor',
                                  bottom=True, left=True)

        ylims_ratio.append(axes_ratio[i].get_ylim())

        this_xlims = axes[i].get_xlim()
        if this_xlims[0] < xlims[0]:
            xlims[0] = this_xlims[0]
        if this_xlims[1] > xlims[1]:
            xlims[1] = this_xlims[1]

        # Label axes
        axes_ratio[i].set_xlabel(r"$L/$ [erg $/$ s $/$ Hz]")

    for i in range(len(axes)):
        axes[i].set_ylim(10 ** -0.7, 10 ** 0.8)
        axes[i].set_xlim(10 ** extent[2], 10 ** 31.3)
        axes_ratio[i].set_xlim(10 ** extent[2], 10 ** 31.)
        axes_ratio[i].tick_params(axis='x', which='both', bottom=True)

    for i in range(len(axes)):
        axes[i].set_ylim(10 ** -0.8, 10**0.8)
        axes_ratio[i].set_ylim(np.min(ylims_ratio), np.max(ylims_ratio))

    axes[0].set_ylabel('$R/ [pkpc]$')
    axes_ratio[0].set_ylabel('$R_{\mathrm{Att}}/ R_{\mathrm{Int}}$')
    axes[0].tick_params(axis='y', which='both', left=True)
    axes_ratio[0].tick_params(axis='y', which='both', left=True)

    uni_legend_elements = []
    for f in filters:
        uni_legend_elements.append(
            Line2D([0], [0], color=cmap(norm(trans[f][1])), linestyle="-",
                   label=f.split(".")[-1]))
    included = []
    for l in legend_elements:
        if (l.get_label(), l.get_marker()) not in included:
            print((l.get_label(), l.get_marker()))
            uni_legend_elements.append(l)
            included.append((l.get_label(), l.get_marker()))
    uni_legend_elements.append(
        Line2D([0], [0], color="k", linestyle="-",
               label="FLARES"))
    uni_legend_elements.append(
        Line2D([0], [0], color="k", linestyle="--",
               label="BlueTides"))

    axes_ratio[2].legend(handles=uni_legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, -0.35), fancybox=True,
                         nrow=2)

    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm)
    cb1.set_label("$\lambda / [\mu\mathrm{m}]$")

    fig.savefig(
        'plots/FilterCompHalfLightRadius_' + mtype + "_" + sample + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight', dpi=300)

    plt.close(fig)
