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
from astropy.convolution import Gaussian2DKernel
import photutils as phut
from photutils import find_peaks
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

cmap = mpl.cm.get_cmap("autumn")
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

if sys.argv[3] == "All":
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
else:
    snaps = sys.argv[3]

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

kernel_sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM = 3
kernel = Gaussian2DKernel(kernel_sigma, x_size=3, y_size=3)
kernel.normalize()

csoft = 0.001802390 / (0.6777) * 1e3

masslim = 10 ** float(sys.argv[4])

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
weight_dict = {}
imgs_dict = {}

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

lumins = []
hlr = []

z_plot = 7.0

for reg, snap in reg_snaps:

    print(reg, snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hdf = h5py.File("data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
    type_group = hdf[Type]
    orientation_group = type_group[orientation]

    hlr_dict.setdefault(snap, {})
    hlr_app_dict.setdefault(snap, {})
    hlr_pix_dict.setdefault(snap, {})
    lumin_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})
    imgs_dict.setdefault(snap, {})

    if z <= 2.8:
        csoft = 0.000474390 / 0.6777 * 1e3
    else:
        csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

    if z != z_plot:
        continue

    single_pix_area = csoft * csoft

    # Define width
    ini_width = 60

    # Compute the resolution
    ini_res = ini_width / csoft
    res = int(np.ceil(ini_res))

    # Compute the new width
    width = csoft * res

    print(width, res)

    single_pixel_area = csoft * csoft

    # Define range and extent for the images
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
    imgextent = [-width / 2, width / 2, -width / 2, width / 2]

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hlr_app_dict[snap].setdefault(f, [])
        hlr_pix_dict[snap].setdefault(f, [])
        lumin_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])
        imgs_dict[snap].setdefault(f, [])

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

        imgs = orientation_group[f]["Images"][...][okinds]

        print(imgs.shape)

        for i_img in range(imgs.shape[0]):

            img = imgs[i_img, :, :]

            img[img < 10**19] = 0

            # threshold = phut.detect_threshold(img, nsigma=5)
            threshold = np.median(img) + np.std(img)

            try:
                segm = phut.detect_sources(img, threshold, npixels=5,
                                           filter_kernel=kernel)
                segm = phut.deblend_sources(img, segm, npixels=5,
                                            filter_kernel=kernel,
                                            nlevels=32, contrast=0.001)
            except TypeError:
                continue
            # x_cent = []
            # y_cent = []
            # for i in range(1, np.max(segm.data) + 1):
            #     test_img = img
            #     test_img[segm.data != i] = 0.0
            #     tbl = find_peaks(test_img, threshold, box_size=5)
            #     print(tbl)
            #     x_cent.append((tbl["x_peak"] - 0.5 - (img.shape[0] / 2.)) * csoft)
            #     y_cent.append((tbl["y_peak"] - 0.5 - (img.shape[0] / 2.)) * csoft)

            for i in range(np.max(segm.data + 1)):
                if np.sum(img[segm.data == i]) < 10**27:
                    continue
                hlr.append(util.get_pixel_hlr(img[segm.data == i],
                                              single_pix_area,
                                              radii_frac=0.5))
                lumins.append(np.sum(img[segm.data == i]))

            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            # ax1.grid(False)
            # ax2.grid(False)
            # ax1.imshow(np.log10(img), extent=imgextent, cmap="Greys_r")
            # ax2.imshow(segm.data, extent=imgextent)
            # circle1 = plt.Circle((0, 0), 30, color='r', fill=False)
            # ax1.add_artist(circle1)
            # circle1 = plt.Circle((0, 0), hlr_app_dict[snap][f][i_img],
            #                      color='g', linestyle="--", fill=False)
            # ax1.add_artist(circle1)
            # circle1 = plt.Circle((0, 0), hlr_dict[snap][f][i_img],
            #                      color='b', linestyle="--", fill=False)
            # ax1.add_artist(circle1)
            # circle1 = plt.Circle((0, 0), hlr_pix_dict[snap][f][i_img],
            #                      color='y', linestyle=":", fill=False)
            # ax1.add_artist(circle1)
            # fig.savefig("plots/gal_img_log_" + f + "_%.1f.png"
            #             % np.log10(np.sum(img)))
            # plt.close(fig)

    hdf.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax1.grid(False)
try:

    cbar = ax.hexbin(np.array(lumins), np.array(hlr), gridsize=50, mincnt=1,
                     xscale='log', yscale='log',
                     norm=LogNorm(), linewidths=0.2,
                     cmap='viridis')
    ax1.hexbin(np.array(lumins), np.array(hlr) * cosmo.arcsec_per_kpc_proper(z).value,
               gridsize=50, mincnt=1, xscale='log',
               yscale='log', norm=LogNorm(), linewidths=0.2,
               cmap='viridis', alpha=0)
except ValueError as e:
    print(e)

legend_elements = []

for p in labels.keys():
    okinds = papers == p
    plt_m = mags[okinds]
    plt_r_es = r_es[okinds]
    plt_zs = zs[okinds]

    okinds = np.logical_and(plt_zs >= (z_plot - 0.5),
                            np.logical_and(plt_zs < (z_plot + 0.5),
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
               marker=markers[p], label=labels[p], s=25,
               color=colors[p], alpha=0.7)

ax1.set_ylabel('$R_{1/2}/ [arcsecond]$')

# Label axes
ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
ax.set_ylabel('$R_{1/2}/ [pkpc]$')

ax.tick_params(axis='x', which='minor', bottom=True)

# ax.set_xlim(10 ** 26.9, 10 ** 30.5)

ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

fig.savefig('plots/HalfLightRadius_mock_detect_test.png', bbox_inches='tight')
