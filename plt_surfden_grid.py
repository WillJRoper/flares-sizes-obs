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
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import h5py
import sys
import pandas as pd

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

# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
         '010_z005p000']

# Define filter
filters = ['FAKE.TH.' + f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]

csoft = 0.001802390 / (0.6777) * 1e3

nlim = 10 ** 8

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
surf_den_dict = {}
img_lumin_dict = {}
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

    try:
        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
                                                        orientation),
            "r")
    except OSError as e:
        print(e)
        continue

    hlr_dict.setdefault(snap, {})
    surf_den_dict.setdefault(snap, {})
    img_lumin_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    if z <= 2.8:
        csoft = 0.000474390 / 0.6777 * 1e3
    else:
        csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

    single_pixel_area = csoft * csoft

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        surf_den_dict[snap].setdefault(f, [])
        img_lumin_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = hdf[f]["Mass"][...]
        okinds = masses > nlim

        print(reg, snap, f, masses[okinds].size)

        img_lumins = hdf[f]["Image_Luminosity"][...][okinds]
        hlrs = hdf[f]["HLR_0.5"][...][okinds]
        masses = masses[okinds]

        # surf_den = hdf[f]["Surface_Density"][...][okinds]
        surf_den = img_lumins / (2 * np.pi * hlrs**2)

        hlr_dict[snap][f].extend(hlrs)
        surf_den_dict[snap][f].extend(surf_den)
        img_lumin_dict[snap][f].extend(img_lumins)
        mass_dict[snap][f].extend(masses)
        weight_dict[snap][f].extend(np.full(masses.size,
                                            weights[int(reg)]))
        #
        # for i in range(masses.size):
        #     img = imgs[i]
        #
        #     surf_den = img_lumins[i] / (
        #             img[img > 0].size * single_pixel_area)
        #
        #     hlr_dict[snap][f].append(hlrs[i])
        #     surf_den_dict[snap][f].append(surf_den)
        #     img_lumin_dict[snap][f].append(img_lumins[i])
        #     mass_dict[snap][f].append(masses[i])
        #     weight_dict[snap][f].append(weights[int(reg)])

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # axes[i].imshow(np.log10(img), extent=imgextent)
        # axes[i].grid(False)
        # circle1 = plt.Circle((0, 0), 30, color='r', fill=False)
        # axes[i].add_artist(circle1)
        # circle1 = plt.Circle((0, 0), hlr_app_dict[tag][f][-1],
        #                      color='g', linestyle="--", fill=False)
        # axes[i].add_artist(circle1)
        # circle1 = plt.Circle((0, 0), hlr_dict[tag][f][-1],
        #                      color='b', linestyle="--", fill=False)
        # axes[i].add_artist(circle1)
        # fig.savefig("plots/gal_img_log_%d.png"
        #             % np.log10(np.sum(this_lumin)))
        # plt.close(fig)

    hdf.close()

for f in filters:

    fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                             np.log10(M_to_lum(-18)),
                             1000)

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, len(snaps))
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    ylims = []
    i = 0
    while i < len(snaps):
        axes.append(fig.add_subplot(gs[0, i]))
        if i > 0:
            axes[-1].tick_params(axis='y', left=False, right=False,
                                 labelleft=False, labelright=False)
        i += 1

    legend_elements = []

    for i, snap in enumerate(snaps):

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_dict[snap][f])
        surf_dens = np.array(surf_den_dict[snap][f])
        masses = np.array(mass_dict[snap][f])

        okinds = np.logical_and(hlrs > 10 ** -2,
                                np.logical_and(surf_dens > 0,
                                               masses > 10 ** 8))

        surf_dens = surf_dens[okinds]
        hlrs = hlrs[okinds]
        masses = masses[okinds]
        w = np.array(weight_dict[snap][f])[okinds]

        try:
            axes[i].hexbin(masses, surf_dens, gridsize=50,
                           mincnt=1, C=w,
                           reduce_C_function=np.sum,
                           xscale='log', yscale='log',
                           norm=LogNorm(), linewidths=0.2,
                           cmap='plasma')
        except ValueError as e:
            print(e)

        ylims.append(axes[i].get_ylim())

        axes[i].set_xlim(10 ** 7.5, 10 ** 12)

    for i in range(len(axes)):
        axes[i].set_ylim(np.min(ylims), np.max(ylims))
        axes[i].set_xlabel("$M_\star/M_\odot$")

    axes[0].set_ylabel(
        "$S / [\mathrm{erg} \mathrm{s}^{-1} \mathrm{Hz}^{-1} \mathrm{Mpc}^{-2}]$")

    fig.savefig(
        'plots/SurfDen_Mass_' + f + '_'
        + orientation + '_' + Type + "_" + extinction + "_"
        + '%d.png' % nlim,
        bbox_inches='tight', dpi=300)

    plt.close(fig)

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, len(snaps))
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    ylims = []
    i = 0
    while i < len(snaps):
        axes.append(fig.add_subplot(gs[0, i]))
        if i > 0:
            axes[-1].tick_params(axis='y', left=False, right=False,
                                 labelleft=False, labelright=False)
        i += 1

    legend_elements = []

    for i, snap in enumerate(snaps):

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_dict[snap][f])
        surf_dens = np.array(surf_den_dict[snap][f])
        masses = np.array(mass_dict[snap][f])

        okinds = np.logical_and(hlrs > 10 ** -2,
                                np.logical_and(surf_dens > 0,
                                               masses > 10 ** 8))

        surf_dens = surf_dens[okinds]
        hlrs = hlrs[okinds]
        masses = masses[okinds]
        w = np.array(weight_dict[snap][f])[okinds]

        try:
            axes[i].hexbin(hlrs, surf_dens, gridsize=50,
                           mincnt=1, C=w,
                           reduce_C_function=np.sum,
                           xscale='log', yscale='log',
                           norm=LogNorm(), linewidths=0.2,
                           cmap='plasma')
        except ValueError as e:
            print(e)

        ylims.append(axes[i].get_ylim())

        axes[i].set_xlim(10 ** -1.5, 10 ** 1.5)

    for i in range(len(axes)):
        axes[i].set_ylim(np.min(ylims), np.max(ylims))
        axes[i].set_xlabel("$R_{1/2}/ [\mathrm{kpc}]$")

    axes[0].set_ylabel(
        "$S / [\mathrm{erg} \mathrm{s}^{-1} \mathrm{Hz}^{-1} \mathrm{Mpc}^{-2}]$")

    fig.savefig(
        'plots/SurfDen_Size_' + f + '_'
        + orientation + '_' + Type + "_" + extinction + "_"
        + '%d.png' % nlim,
        bbox_inches='tight', dpi=300)

    plt.close(fig)

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, len(snaps))
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    ylims = []
    i = 0
    while i < len(snaps):
        axes.append(fig.add_subplot(gs[0, i]))
        if i > 0:
            axes[-1].tick_params(axis='y', left=False, right=False,
                                 labelleft=False, labelright=False)
        i += 1

    legend_elements = []

    for i, snap in enumerate(snaps):

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        hlrs = np.array(hlr_dict[snap][f])
        surf_dens = np.array(surf_den_dict[snap][f])
        masses = np.array(mass_dict[snap][f])

        okinds = np.logical_and(hlrs > 10 ** -2,
                                np.logical_and(surf_dens > 0,
                                               masses > 10 ** 8))

        surf_dens = surf_dens[okinds]
        hlrs = hlrs[okinds]
        masses = masses[okinds]
        w = np.array(weight_dict[snap][f])[okinds]

        try:
            im = axes[i].hexbin(masses, hlrs, gridsize=50,
                                mincnt=1, C=surf_dens,
                                reduce_C_function=np.mean,
                                xscale='log', yscale='log',
                                norm=LogNorm(vmin=10 ** 24, vmax=10 ** 31),
                                linewidths=0.2,
                                cmap='plasma')
        except ValueError as e:
            print(e)

        ylims.append(axes[i].get_ylim())

    for i in range(len(axes)):
        axes[i].set_ylim(10 ** -1.5, 10 ** 1.5)
        axes[i].set_xlim(10 ** 7.5, 10 ** 12)
        axes[i].set_xlabel("$M_\star/M_\odot$")

    axes[0].set_ylabel("$R_{1/2}/ [\mathrm{kpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$S / [\mathrm{erg} \mathrm{s}^{-1} \mathrm{Hz}^{-1} \mathrm{Mpc}^{-2}]$")

    fig.savefig(
        'plots/SurfDen_MassSize_' + f + '_'
        + orientation + '_' + Type + "_" + extinction + "_"
        + '%d.png' % nlim,
        bbox_inches='tight', dpi=300)

    plt.close(fig)
