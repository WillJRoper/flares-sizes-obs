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
import cmasher as cmr
import utilities as util
from flare import plt as flareplt

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
    return I0 * np.exp(-(r / r0))


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

# reg, snap = regions[0], '010_z005p000'

np.random.seed(100)

# for snap in snaps:
#
#     reg = regions[0]
#
#     hdf = h5py.File(
#         "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
#                                                     orientation),
#         "r")
#
#     f = filters[0]
#     img_shape = hdf[f]["Images"].shape[-1]
#
#     hdf.close()
#
#     dpi = img_shape
#     fig = plt.figure(figsize=(4, len(filters)), dpi=dpi)
#     fig_log = plt.figure(figsize=(4, len(filters)), dpi=dpi)
#     gs = gridspec.GridSpec(ncols=4, nrows=len(filters))
#     gs.update(wspace=0.0, hspace=0.0)
#     axes = np.empty((len(filters), 4), dtype=object)
#     axes_log = np.empty((len(filters), 4), dtype=object)
#     bins = [10 ** 8, 10 ** 9, 10 ** 9.5, 10 ** 10, np.inf]
#
#     stacks = {}
#     num_stacked = {}
#
#     for f in filters:
#
#         stacks[f] = {}
#         num_stacked[f] = {}
#         for b in bins[:-1]:
#             stacks[f][b] = np.zeros((img_shape, img_shape))
#             num_stacked[f][b] = 0
#
#         for reg in regions:
#
#             print(snap, f, reg)
#
#             reg = regions[0]
#
#             hdf = h5py.File(
#                 "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap,
#                                                             Type,
#                                                             orientation),
#                 "r")
#
#             for i, b in enumerate(bins[:-1]):
#                 masses = hdf[f]["Mass"][...]
#                 okinds = np.logical_and(masses >= b,
#                                         masses < bins[i + 1])
#                 imgs = hdf[f]["Images"][...]
#                 stacks[f][b] += np.sum(imgs[okinds, :, :],
#                                        axis=0)
#
#                 num_stacked[f][b] += masses[okinds].size
#
#             hdf.close()
#
#     z_str = snap.split('z')[1].split('p')
#     z = float(z_str[0] + '.' + z_str[1])
#
#     all_imgs = []
#     for f in filters:
#         for b in bins[:-1]:
#             stacks[f][b] = stacks[f][b] / num_stacked[f][b]
#             all_imgs.append(stacks[f][b])
#     all_imgs = np.array(all_imgs)
#     print(all_imgs.shape)
#     norm = cm.Normalize(vmin=0,
#                         vmax=np.percentile(all_imgs[all_imgs > 0], 99.99),
#                         clip=True)
#     norm_log = cm.LogNorm(vmin=0,
#                           vmax=np.percentile(all_imgs[all_imgs > 0], 99),
#                           clip=True)
#
#     for i in range(len(filters)):
#         for j in range(4):
#             axes[i, j] = fig.add_subplot(gs[i, j])
#             axes_log[i, j] = fig_log.add_subplot(gs[i, j])
#
#             # Remove axis labels and ticks
#             axes[i, j].tick_params(axis='x', top=False, bottom=False,
#                                    labeltop=False, labelbottom=False)
#             axes[i, j].tick_params(axis='y', left=False, right=False,
#                                    labelleft=False, labelright=False)
#
#             # Remove axis labels and ticks
#             axes_log[i, j].tick_params(axis='x', top=False,
#                                        bottom=False,
#                                        labeltop=False,
#                                        labelbottom=False)
#             axes_log[i, j].tick_params(axis='y', left=False,
#                                        right=False,
#                                        labelleft=False,
#                                        labelright=False)
#
#     for j, b in enumerate(bins[:-1]):
#         for i, f in enumerate(filters):
#
#             if j == 0:
#                 axes[i, j].set_ylabel(f.split(".")[-1], fontsize=6)
#                 axes_log[i, j].set_ylabel(f.split(".")[-1], fontsize=6)
#             if i == 0:
#                 if bins[j + 1] == np.inf:
#                     axes[i, j].set_title("%d $\leq "
#                                          "\log_{10}(M/M_\odot)$"
#                                          % (np.log10(bins[j])), fontsize=6)
#                     axes_log[i, j].set_title("%d $\leq "
#                                              "\log_{10}(M/M_\odot)$"
#                                              % (np.log10(bins[j])), fontsize=6)
#                 else:
#                     axes[i, j].set_title("%d $\leq "
#                                          "\log_{10}(M/M_\odot) <$ %d"
#                                          % (np.log10(bins[j]),
#                                             np.log10(bins[j + 1])), fontsize=6)
#                     axes_log[i, j].set_title("%d $\leq "
#                                          "\log_{10}(M/M_\odot) <$ %d"
#                                          % (np.log10(bins[j]),
#                                             np.log10(bins[j + 1])), fontsize=6)
#
#             size = stacks[f][b].shape[0]
#             axes[i, j].imshow(stacks[f][b][int(0.4 * size):-int(0.4 * size),
#                               int(0.4 * size):-int(0.4 * size)],
#                               cmap=cmr.neutral_r)
#
#             axes_log[i, j].imshow(
#                 stacks[f][b][int(0.3 * size):-int(0.3 * size),
#                 int(0.3 * size):-int(0.3 * size)],
#                 cmap=cmr.neutral, norm=cm.LogNorm())
#
#     fig.savefig(
#         'plots/Image_grids/StackImgGrid_' + reg
#         + '_' + snap + '_' + orientation + '_' + Type
#         + "_" + extinction + "".replace(".", "p") + ".pdf",
#         bbox_inches='tight', dpi=fig.dpi)
#     fig_log.savefig(
#         'plots/Image_grids/StackLogImgGrid_' + reg
#         + '_' + snap + '_' + orientation + '_' + Type
#         + "_" + extinction + "".replace(".", "p") + ".pdf",
#         bbox_inches='tight', dpi=fig_log.dpi)
#     plt.close(fig)

for f in filters:

    row_filters = [f, ]

    imgs_dict = {}
    hlr_pix_dict = {}
    lumin_dict = {}
    mass_dict = {}
    sd_dict = {}
    hlr_dict = {}
    hlr_err_dict = {}

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
        bin_wid = 10**0.2
        bins = np.arange(8, 11.8, np.log10(bin_wid))
        bins = np.array([10**b for b in bins])
        bin_cents = (bins[1:] + bins[:-1]) / 2

        stacks = {}
        num_stacked = {}

        for f in row_filters:

            reg = regions[0]

            hdf = h5py.File(
                "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(
                    reg, snap,
                    Type,
                    orientation,
                    f.split(
                        ".")[
                        -1]),
                "r")

            f = row_filters[0]
            img_shape = hdf[f]["Images"].shape[-1]

            hdf.close()

            stacks[f] = {}
            num_stacked[f] = {}
            for b in bins[:-1]:
                stacks[f][b] = np.zeros((img_shape, img_shape))
                num_stacked[f][b] = 0

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
                    masses = hdf[f]["Mass"][...]
                    okinds = np.logical_and(masses >= b,
                                            masses < bins[i + 1])
                    imgs = hdf[f]["Images"][...]
                    stacks[f][b] += np.sum(imgs[okinds, :, :], axis=0)

                    num_stacked[f][b] += masses[okinds].size

                    hlr_dict.setdefault(b, []).extend(hlrs[okinds])

                hdf.close()

        all_imgs = []
        for f in row_filters:
            for b in bins[:-1]:
                stacks[f][b] = stacks[f][b] / num_stacked[f][b]
                all_imgs.append(stacks[f][b])
        all_imgs = np.array(all_imgs)
        print(all_imgs.shape)

        # Compute mean and standard error for half light radii
        mean_hlrs = np.zeros(len(bins[:-1]))
        serr_hlrs = np.zeros(len(bins[:-1]))
        for ind, b in enumerate(bins[:-1]):
            mean_hlrs[ind] = np.mean(hlr_dict[b])
            serr_hlrs[ind] = np.std(hlr_dict[b])

        # Define list to store profiles
        stack_hlrs = np.zeros(len(bins[:-1]))
        stack_scale_lengths = np.zeros(len(bins[:-1]))
        stack_sl_errs = np.zeros(len(bins[:-1]))

        for f in filters:
            for j, b in enumerate(bins[:-1]):

                # Extract image
                plt_img = stacks[f][b]

                # Calculate image half light radius
                hlr = util.get_pixel_hlr(stacks[f][b], single_pixel_area,
                                         0.5)
                stack_hlrs[j] = hlr

                # Calculate a plot 1D profiles
                xs = np.linspace(-(plt_img.shape[0] / 2) * csoft,
                                 (plt_img.shape[0] / 2) * csoft,
                                 plt_img.shape[0])
                ys = np.sum(plt_img, axis=0) / np.sum(plt_img)

                tot_lum = np.sum(stacks[f][b])
                popt, pcov = curve_fit(exp_fit, xs, ys,
                                       p0=(
                                           tot_lum * 0.2,
                                           0.2))

                print(b, "I_0=", popt[0], "+/-", np.sqrt(pcov[0, 0]))
                print(b, "r_0=", popt[1], "+/-", np.sqrt(pcov[1, 1]))
                print(b, "R_1/2=", hlr)

                # Store scale lengths
                stack_scale_lengths[j] = popt[1]
                stack_sl_errs[j] = np.sqrt(pcov[1, 1])

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.loglog()

        # Plot effective half light radii and scale length
        ax.plot(bin_cents, stack_hlrs, color="k", linestyle="-")
        ax.errorbar(bin_cents, stack_scale_lengths, yerr=stack_sl_errs,
                    xerr=bin_wid, capsize=5, marker="s", linestyle="none")

        ax.set_ylabel("$R_{1/2} / [\mathrm{pkpc}]$")
        ax.set_xlabel("$M_\star / M_\odot$")

        fig.savefig(
            'plots/' + str(z) + '/ScaleLengthComp_' + f + '_' + snap
            + '_' + orientation + '_' + Type
            + "_" + extinction + "".replace(".", "p") + ".pdf",
            bbox_inches='tight', dpi=fig_log.dpi)
        plt.close(fig)
