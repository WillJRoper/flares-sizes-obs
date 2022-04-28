#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import flare.photom as photconv
from scipy.optimize import curve_fit
import h5py
import sys
import pandas as pd
import utilities as util
from matplotlib.colors import LogNorm
from flare import plt as flareplt


# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

# Set plotting fontsizes
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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


def exp_fit(r, I0, r0):
    return I0 * np.exp(-(np.abs(r) / r0))


# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
#                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
filters = ['FAKE.TH.' + f
           for f in ['FUV', ]]

csoft = 0.001802390 / (0.6777) * 1e3

imgs_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
sd_dict = {}

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['010_z005p000', ]
# snaps = ['007_z008p000', '010_z005p000']

np.random.seed(100)

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

# Define the norm
weight_norm = LogNorm(vmin=10 ** -4, vmax=1)

# Define extent
size_tot_extent = [-1.1, 1.3, 7.7, 11.3]

for f in filters:

    row_filters = [f, ]

    imgs_dict = {}
    hlr_pix_dict = {}
    lumin_dict = {}
    mass_dict = {}
    sd_dict = {}
    all_hlrs = []
    all_mass = []
    hlr_err_dict = {}
    w_dict = {}

    np.random.seed(100)

    for snap in snaps:

        # Get redshift
        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        # Getting softening length
        if z <= 2.8:
            csoft = 0.000474390 / 0.6777 * 1e3
        else:
            csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

        # Define width
        ini_width = 60

        # Compute the resolution
        ini_res = ini_width / (csoft)
        res = int(np.ceil(ini_res))

        # Compute the new width
        width = csoft * res

        print("Image width and resolution", width, res)

        # Compute pixel area
        single_pixel_area = csoft * csoft

        profile_lims = [1, 0]

        # Define mass bins
        bin_wid = 10**0.4
        bins = np.arange(8, 12.5, np.log10(bin_wid))
        bins = np.array([10**b for b in bins])
        bin_cents = (bins[1:] + bins[:-1]) / 2

        stacks = {}
        num_stacked = {}

        for f in row_filters:

            stacks[f] = {}
            num_stacked[f] = {}

            for reg in regions:

                print(snap, f, reg)

                reg = regions[0]

                hdf = h5py.File(
                    "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(
                        reg,
                        snap,
                        Type,
                        orientation,
                        f.split(
                            ".")[
                            -1]),
                    "r")

                for i, b in enumerate(bins[:-1]):

                    hlrs = hdf[f]["HLR_0.5"][...]
                    okinds = hlrs > 0
                    hlrs = hlrs[okinds]
                    masses = hdf[f]["Mass"][...]
                    masses = masses[okinds]
                    imgs = hdf[f]["Images"][...]
                    imgs = imgs[okinds, :, :]
                    okinds = np.logical_and(masses >= b,
                                            masses < bins[i + 1])
                    img_shape = imgs[0, :, :].shape
                    stacks[f].setdefault(b, np.zeros(img_shape))
                    num_stacked[f].setdefault(b, 0)
                    stacks[f][b] += np.sum(imgs[okinds, :, :], axis=0)

                    num_stacked[f][b] += masses[okinds].size

                    all_mass.extend(masses[okinds])
                    all_hlrs.extend(hlrs[okinds])
                    w_dict.setdefault(b, []).extend(np.full(hlrs[okinds].size,
                                                            weights[int(reg)]))

                hdf.close()

        all_imgs = []
        for f in row_filters:
            for b in bins[:-1]:
                stacks[f][b] = stacks[f][b] / num_stacked[f][b]
                all_imgs.append(stacks[f][b])
        all_imgs = np.array(all_imgs)
        print(all_imgs.shape)

        # Initialise mean and standard error for half light radii
        mean_hlrs = np.zeros(len(bins[:-1]))
        serr_hlrs = np.zeros(len(bins[:-1]))

        # Define list to store profiles
        stack_hlrs = np.zeros(len(bins[:-1]))
        stack_scale_lengths = np.zeros(len(bins[:-1]))
        stack_sl_errs = np.zeros(len(bins[:-1]))

        for f in filters:
            for j, b in enumerate(bins[:-1]):

                # if not len(hlr_dict[b]) > 0:
                #     continue

                # Extract image
                plt_img = stacks[f][b]

                # Calculate image half light radius
                hlr = util.get_pixel_hlr(plt_img, single_pixel_area,
                                         0.5)
                stack_hlrs[j] = hlr

                # Calculate a plot 1D profiles
                xs = np.linspace(-(plt_img.shape[0] / 2) * csoft,
                                 (plt_img.shape[0] / 2) * csoft,
                                 plt_img.shape[0])
                ys = np.nansum(plt_img, axis=0)

                # mean_hlrs[j] = np.average(hlr_dict[b], weights=w_dict[b])
                # serr_hlrs[j] = np.std(hlr_dict[b]) / np.sqrt(len(hlr_dict[b]))

                tot_lum = np.nansum(plt_img)
                popt, pcov = curve_fit(exp_fit, xs, ys,
                                       p0=(tot_lum * 0.2, 1))

                print(b, "I_0=", popt[0], "+/-", np.sqrt(pcov[0, 0]))
                print(b, "r_0=", popt[1], "+/-", np.sqrt(pcov[1, 1]))
                print(b, "R_1/2=", hlr)

                # Store scale lengths
                stack_scale_lengths[j] = popt[1]
                stack_sl_errs[j] = np.sqrt(pcov[1, 1])

        # Initialise file to save outputs
        hdf_out = h5py.File("plots/flares-sizes-results.hdf5", "r")

        com_comp = hdf_out[snap][f]["Intrinsic"]["Compact_Population_Complete"][...]
        diff_comp = hdf_out[snap][f]["Intrinsic"]["Diffuse_Population_Complete"][...]
        mass = hdf_out[snap][f]["Intrinsic"]["Mass"][...]
        hlr = hdf_out[snap][f]["Intrinsic"]["HLR_0.5"][...]
        w = hdf_out[snap][f]["Intrinsic"]["Weight"][...]

        fig = plt.figure(figsize=(1.4 * 3.5, 1.2 * 3.5))
        gs = gridspec.GridSpec(1, 3, width_ratios=(10, 1, 1))
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        cax2 = fig.add_subplot(gs[0, 2])
        ax.semilogx()

        # Plot effective half light radii and scale length
        cbar = ax.hexbin(mass[diff_comp], hlrs[diff_comp],
                         C=w[diff_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='Greys',
                         extent=[extent[2], extent[3], extent[0],
                                 extent[1]])
        cbar = ax.hexbin(mass[com_comp], hlrs[com_comp],
                         C=w[com_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=[extent[2], extent[3],
                                                 extent[0], extent[1]])
        okinds = stack_scale_lengths > 0
        ax.errorbar(bin_cents[okinds], stack_scale_lengths[okinds],
                    yerr=stack_sl_errs[okinds],
                    capsize=5, color="k", marker="s", linestyle="--",
                    markersize=3, label=r"$R_{\mathrm{exp}}$")

        ax.set_ylabel(r"$R / [\mathrm{pkpc}]$")
        ax.set_xlabel(r"$M_\star / M_\odot$")

        ax.legend(loc="best")

        fig.savefig(
            'plots/' + str(z) + '/ScaleLengthComp_' + f + '_' + snap
            + '_' + orientation + '_' + Type
            + "_" + extinction + "".replace(".", "p") + ".pdf",
            bbox_inches='tight')
        plt.close(fig)

