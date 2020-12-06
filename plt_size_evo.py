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

cmap = mpl.cm.get_cmap("jet")
norm = plt.Normalize(vmin=0, vmax=1)

labels = {"C16": "Calvi+2016",
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
for key, col in zip(labels.keys(), np.linspace(0, 1, len(labels.keys()))):
    colors[key] = cmap(norm(col))


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ https://stackoverflow.com/questions/21844024/
        weighted-percentile-using-numpy

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


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

# Define luminosity and dust model types
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

    hdf = h5py.File("data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
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

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Filter =", f)

    hlr_med = []
    hlr_16 = []
    hlr_84 = []
    hlr_med_app = []
    hlr_16_app = []
    hlr_84_app = []
    hlr_med_pix = []
    hlr_16_pix = []
    hlr_84_pix = []
    intr_hlr_med = []
    intr_hlr_16 = []
    intr_hlr_84 = []
    intr_hlr_med_app = []
    intr_hlr_16_app = []
    intr_hlr_84_app = []
    intr_hlr_med_pix = []
    intr_hlr_16_pix = []
    intr_hlr_84_pix = []
    plt_z = []

    for snap in snaps:

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_dict[snap][f])
        intr_hlrs = np.array(intr_hlr_dict[snap][f])
        hlrs_app = np.array(hlr_app_dict[snap][f])
        intr_hlrs_app = np.array(intr_hlr_app_dict[snap][f])
        hlrs_pix = np.array(hlr_pix_dict[snap][f])
        intr_hlrs_pix = np.array(intr_hlr_pix_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        intr_lumins = np.array(intr_lumin_dict[snap][f])
        w = np.array(weight_dict[snap][f])

        if len(w) == 0:
            continue

        quants = weighted_quantile(hlrs, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        hlr_med.append(quants[1])
        hlr_16.append(quants[0])
        hlr_84.append(quants[2])

        quants = weighted_quantile(hlrs_app, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        hlr_med_app.append(quants[1])
        hlr_16_app.append(quants[0])
        hlr_84_app.append(quants[2])

        quants = weighted_quantile(hlrs_pix, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        hlr_med_pix.append(quants[1])
        hlr_16_pix.append(quants[0])
        hlr_84_pix.append(quants[2])

        quants = weighted_quantile(intr_hlrs, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        intr_hlr_med.append(quants[1])
        intr_hlr_16.append(quants[0])
        intr_hlr_84.append(quants[2])

        quants = weighted_quantile(intr_hlrs_app, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        intr_hlr_med_app.append(quants[1])
        intr_hlr_16_app.append(quants[0])
        intr_hlr_84_app.append(quants[2])

        quants = weighted_quantile(intr_hlrs_pix, [0.16, 0.5, 0.84], sample_weight=w,
                                   values_sorted=False, old_style=False)

        intr_hlr_med_pix.append(quants[1])
        intr_hlr_16_pix.append(quants[0])
        intr_hlr_84_pix.append(quants[2])

        plt_z.append(z)

    soft = []
    for z in plt_z:

        if z <= 2.8:
            soft.append(0.000474390 / 0.6777 * 1e3)
        else:
            soft.append(0.001802390 / (0.6777 * (1 + z)) * 1e3)

    legend_elements = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy()
    ax.plot(plt_z, soft, color="k", linestyle="--", label="Softening")
    try:
        ax.fill_between(plt_z, intr_hlr_16, intr_hlr_84, color="r", alpha=0.4)
        ax.plot(plt_z, intr_hlr_med, color="r", marker="D", linestyle="--")
        legend_elements.append(
            Line2D([0], [0], color="r", linestyle="--", label="Intrinsic"))

        ax.fill_between(plt_z, hlr_16, hlr_84, color="g", alpha=0.4)
        ax.plot(plt_z, hlr_med, color="g", marker="^", linestyle="-")
        legend_elements.append(
            Line2D([0], [0], color="g", linestyle="-",
                   label="Attenuated"))
    except ValueError as e:
        print(e)
        continue

    for p in labels.keys():

        okinds = papers == p
        plt_r_es = r_es[okinds]
        plt_zs = zs[okinds]

        if plt_zs.size == 0:
            continue

        legend_elements.append(
            Line2D([0], [0], marker=markers[p], color='w',
                   label=labels[p], markerfacecolor=colors[p],
                   markersize=8, alpha=0.7))

        ax.scatter(plt_zs, plt_r_es,
                   marker=markers[p], label=labels[p], s=17,
                   color=colors[p], alpha=0.7)

    # Label axes
    ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    ax.set_ylabel('$R_{1/2}/ [pkpc]$')

    ax.tick_params(axis='x', which='minor', bottom=True)

    ax.set_xlim(4.5, 11.5)
    ax.set_ylim(10**-1.25, 10**0.8)

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

    fig.savefig(
        'plots/HalfLightRadius_evolution__' + f + '_'
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
    ax.semilogy()
    try:
        ax.fill_between(plt_z, intr_hlr_16_app, intr_hlr_84_app, color="r",
                        alpha=0.4)
        ax.plot(plt_z, intr_hlr_med_app, color="r", marker="D", linestyle="--")
        legend_elements.append(
            Line2D([0], [0], color="B", linestyle="--", label="Intrinsic"))

        ax.fill_between(plt_z, hlr_16_app, hlr_84_app, color="g", alpha=0.4)
        ax.plot(plt_z, hlr_med_app, color="g", marker="^", linestyle="-")
        legend_elements.append(
            Line2D([0], [0], color="g", linestyle="-",
                   label="Attenuated"))
    except ValueError as e:
        print(e)
        continue

    for p in labels.keys():

        okinds = papers == p
        plt_r_es = r_es[okinds]
        plt_zs = zs[okinds]

        if plt_zs.size == 0:
            continue

        legend_elements.append(
            Line2D([0], [0], marker=markers[p], color='w',
                   label=labels[p], markerfacecolor=colors[p],
                   markersize=8, alpha=0.7))

        ax.scatter(plt_zs, plt_r_es,
                   marker=markers[p], label=labels[p], s=17,
                   color=colors[p], alpha=0.7)

    # Label axes
    ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    ax.set_ylabel('$R_{1/2}/ [pkpc]$')

    ax.tick_params(axis='x', which='minor', bottom=True)

    ax.set_xlim(4.5, 11.5)
    ax.set_ylim(10**-1.25, 10**0.8)

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

    fig.savefig('plots/HalfLightRadius_evolution_Aperture_'
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
    ax.semilogy()
    try:
        ax.fill_between(plt_z, intr_hlr_16_pix, intr_hlr_84_pix, color="r", alpha=0.4)
        ax.plot(plt_z, intr_hlr_med_pix, color="r", marker="D", linestyle="--")
        legend_elements.append(
            Line2D([0], [0], color="B", linestyle="--", label="Intrinsic"))

        ax.fill_between(plt_z, hlr_16_pix, hlr_84_pix, color="g", alpha=0.4)
        ax.plot(plt_z, hlr_med_pix, color="g", marker="^", linestyle="-")
        legend_elements.append(
            Line2D([0], [0], color="g", linestyle="-",
                   label="Attenuated"))
    except ValueError as e:
        print(e)
        continue

    for p in labels.keys():

        okinds = papers == p
        plt_r_es = r_es[okinds]
        plt_zs = zs[okinds]
        plt_m = mags[okinds]

        if plt_zs.size == 0:
            continue

        legend_elements.append(
            Line2D([0], [0], marker=markers[p], color='w',
                   label=labels[p], markerfacecolor=colors[p],
                   markersize=8, alpha=0.7))

        ax.scatter(plt_zs, plt_r_es,
                   marker=markers[p], label=labels[p], s=17,
                   color=colors[p], alpha=0.7)

    # Label axes
    ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
    ax.set_ylabel('$R_{1/2}/ [pkpc]$')

    ax.tick_params(axis='x', which='minor', bottom=True)

    ax.set_xlim(4.5, 11.5)
    ax.set_ylim(10**-1.25, 10**0.8)

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

    fig.savefig('plots/HalfLightRadius_evolution_Pixel_'
                + f + '_' + orientation
                + "_" + extinction + "_"
                + '%d.png' % nlim,
                bbox_inches='tight')

    plt.close(fig)
