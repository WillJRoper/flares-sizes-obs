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
import h5py
import sys
import cmasher as cmr
from flare import plt as flareplt


# Set plotting fontsizes
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
Type = sys.argv[2]
extinction = 'default'

filters = ['FAKE.TH.' + f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV']]

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

snaps = ['007_z008p000', '010_z005p000']

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

    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    snaps = ['007_z008p000', '010_z005p000']

    # reg, snap = regions[0], '010_z005p000'

    np.random.seed(100)

    for snap in snaps:

        profile_lims = [1, 0]

        for f in row_filters:

            reg = regions[0]

            hdf = h5py.File(
                "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                   Type,
                                                                   orientation,
                                                                   f.split(
                                                                       ".")[
                                                                       -1]),
                "r")

            f = row_filters[0]
            img_shape = hdf[f]["Images"].shape[-1]

            hdf.close()

            dpi = 300
            fig = plt.figure(dpi=dpi)
            fig_log = plt.figure(dpi=dpi)
            gs = gridspec.GridSpec(ncols=4, nrows=len(row_filters) + 1, height_ratios=[10, 4])
            gs.update(wspace=0.0, hspace=-0.35)
            axes = np.empty((len(row_filters) + 1, 4), dtype=object)
            axes_log = np.empty((len(row_filters) + 1, 4), dtype=object)
            bins = [10 ** 8, 10 ** 9, 10 ** 9.5, 10 ** 10, np.inf]

            stacks = {}
            num_stacked = {}

            stacks[f] = {}
            num_stacked[f] = {}
            for b in bins[:-1]:
                stacks[f][b] = np.zeros((img_shape, img_shape))
                num_stacked[f][b] = 0

            for reg in regions:

                print(snap, f, reg)

                reg = regions[0]

                hdf = h5py.File(
                    "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg,
                                                                       snap,
                                                                       Type,
                                                                       orientation,
                                                                       f.split(
                                                                           ".")[
                                                                           -1]),
                    "r")

                for i, b in enumerate(bins[:-1]):
                    masses = hdf[f]["Mass"][...]
                    okinds = np.logical_and(masses >= b,
                                            masses < bins[i + 1])
                    imgs = hdf[f]["Images"][...]
                    stacks[f][b] += np.sum(imgs[okinds, :, :],
                                           axis=0)

                    num_stacked[f][b] += masses[okinds].size

                hdf.close()

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        all_imgs = []
        for f in row_filters:
            for b in bins[:-1]:
                stacks[f][b] = stacks[f][b] / num_stacked[f][b]
                all_imgs.append(stacks[f][b])
        all_imgs = np.array(all_imgs)
        print(all_imgs.shape)
        norm = cm.Normalize(vmin=0,
                            vmax=np.percentile(all_imgs[all_imgs > 0], 99.99),
                            clip=True)
        norm_log = cm.LogNorm(vmin=np.percentile(all_imgs[all_imgs > 0], 24),
                              vmax=np.percentile(all_imgs[all_imgs > 0], 99.999),
                              clip=True)

        for i in range(len(row_filters)):
            for j in range(4):
                axes[i, j] = fig.add_subplot(gs[i, j])
                axes_log[i, j] = fig_log.add_subplot(gs[i, j])
                axes[i + 1, j] = fig.add_subplot(gs[i + 1, j])
                axes_log[i + 1, j] = fig_log.add_subplot(gs[i + 1, j])

                axes[i + 1, j].grid(True)
                axes_log[i + 1, j].grid(True)

                # Remove axis labels and ticks
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
                axes_log[i, j].tick_params(axis='x', top=False,
                                           bottom=False,
                                           labeltop=False,
                                           labelbottom=False)
                axes_log[i, j].tick_params(axis='y', left=False,
                                           right=False,
                                           labelleft=False,
                                           labelright=False)

                if j > 0:
                    # Remove axis labels and ticks
                    axes[i + 1, j].tick_params(axis='y', left=False, right=False,
                                           labelleft=False, labelright=False)
                    axes_log[i + 1, j].tick_params(axis='y', left=False,
                                               right=False,
                                               labelleft=False,
                                               labelright=False)

        for j, b in enumerate(bins[:-1]):
            for i, f in enumerate(row_filters):

                # if j == 0:
                #     axes[i, j].set_ylabel(f.split(".")[-1], fontsize=6)
                #     axes_log[i, j].set_ylabel(f.split(".")[-1], fontsize=6)
                if i == 0:
                    if bins[j + 1] == np.inf:

                        axes[i, j].set_title("$%.1f \leq \log_{10}(M/M_\odot)$" % (np.log10(bins[j])), fontsize=8)

                        axes_log[i, j].set_title("$%.1f \leq \log_{10}(M/M_\odot)$" % (np.log10(bins[j])), fontsize=8)
                    else:
                        axes[i, j].set_title("$%.1f \leq \log_{10}(M/M_\odot) < %.1f$" % (np.log10(bins[j]), np.log10(bins[j + 1])), fontsize=8)

                        axes_log[i, j].set_title("$%.1f \leq \log_{10}(M/M_\odot) < %.1f$" % (np.log10(bins[j]), np.log10(bins[j + 1])), fontsize=8)

                # Get softening length
                if z <= 2.8:
                    csoft = 0.000474390 / 0.6777 * 1e3
                else:
                    csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

                # Extract central region of image
                size = stacks[f][b].shape[0]
                plt_img = stacks[f][b][int(0.3 * size):-int(0.3 * size),
                          int(0.3 * size):-int(0.3 * size)]
                extent = [0, plt_img.shape[0] * csoft,
                          0, plt_img.shape[1] * csoft]

                # Plot images
                axes[i, j].imshow(plt_img, cmap=cmr.neutral_r, extent=extent)
                axes_log[i, j].imshow(plt_img, cmap=cmr.neutral,
                                      norm=cm.LogNorm(vmin=np.percentile(plt_img, 16)),
                                      extent=extent)

                # Calculate a plot 1D profiles
                xs = np.linspace(-(plt_img.shape[0] / 2) * csoft, (plt_img.shape[0] / 2) * csoft, plt_img.shape[0])
                ys = np.sum(plt_img, axis=0) / np.sum(plt_img)
                axes[i + 1, j].semilogy(xs, ys, color="r")
                axes_log[i + 1, j].semilogy(xs, ys, color="r")

                ylims = axes[i + 1, j].get_ylim()
                if ylims[0] < profile_lims[0]:
                    profile_lims[0] = ylims[0]
                if ylims[1] > profile_lims[1]:
                    profile_lims[1] = ylims[1]

                axes[i + 1, j].set_xlabel("$x / [\mathrm{pkpc}]$")
                axes_log[i + 1, j].set_xlabel("$x / [\mathrm{pkpc}]$")

                print("Image size:",
                      csoft *
                      stacks[f][b][int(0.3 * size):-int(0.3 * size),
                      int(0.3 * size):-int(0.3 * size)].shape[0], "pkpc")

        for j, b in enumerate(bins[:-1]):
            axes[1, j].set_ylim(profile_lims[0], profile_lims[1])
            axes_log[1, j].set_ylim(profile_lims[0], profile_lims[1])

        axes[1, 0].set_ylabel(r"$L_{\mathrm{FUV}}/\Sigma L_{\mathrm{FUV}}$")
        axes_log[1, 0].set_ylabel(r"$L_{\mathrm{FUV}}/\Sigma L_{\mathrm{FUV}}$")

        fig.savefig(
            'plots/Image_grids/StackImgRow_' + f + '_' + reg
            + '_' + snap + '_' + orientation + '_' + Type
            + "_" + extinction + "".replace(".", "p") + ".pdf",
            bbox_inches='tight', dpi=fig.dpi)
        fig_log.savefig(
            'plots/Image_grids/StackLogImgRow_' + f + '_' + reg
            + '_' + snap + '_' + orientation + '_' + Type
            + "_" + extinction + "".replace(".", "p") + ".pdf",
            bbox_inches='tight', dpi=fig_log.dpi)
        plt.close(fig)
