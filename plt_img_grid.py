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

# Define filter
# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
#                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV']]
#                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
filters = ['FAKE.TH.' + f
           for f in ['FUV', ]]

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

for reg in regions:
    for snap in snaps:
        for f in filters:

            hdf = h5py.File(
                "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                   "Total",
                                                                   orientation,
                                                                   f.split(
                                                                       ".")[
                                                                       -1]),
                "r")

            imgs_dict[f] = hdf[f]["Images"][...]
            mass_dict[f] = hdf[f]["Mass"][...]
            hlr_pix_dict[f] = hdf[f]["HLR_Pixel_0.5"][...]
            lumin_dict[f] = hdf[f]["Luminosity"][...]
            sd_dict[f] = hdf[f]["Image_Luminosity"][...] / (
                    2 * np.pi * hdf[f]["HLR_Pixel_0.5"][...] ** 2)

        hdf.close()

        for f in filters:

            print("Plotting for:")
            print("Region = ", reg)
            print("Snapshot = ", snap)
            print("Orientation =", orientation)
            print("Type =", Type)
            print("Filter =", f)

            legend_elements = []

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

            imgs = np.array(imgs_dict[f])
            hlrs_pix = np.array(hlr_pix_dict[f])
            lumins = np.array(lumin_dict[f])
            mass = np.array(mass_dict[f])
            sd = np.array(sd_dict[f])

            print(imgs.shape)

            norm = cm.Normalize(vmin=np.percentile(imgs[imgs > 0], 16),
                                vmax=np.percentile(imgs[imgs > 0], 99.99),
                                clip=True)
            norm_log = cm.Normalize(vmin=np.percentile(
                                        np.log10(imgs[imgs > 0]), 16),
                                    vmax=np.percentile(
                                        np.log10(imgs[imgs > 0]), 99.99),
                                    clip=True)

            print(np.min(imgs[imgs > 0]),
                  np.percentile(imgs[imgs > 0], 33.175),
                  np.percentile(imgs[imgs > 0], 50),
                  np.percentile(imgs[imgs > 0], 99))

            dpi = 1080
            fig = plt.figure(figsize=(4, 4), dpi=dpi)
            fig_log = plt.figure(figsize=(4, 4), dpi=dpi)
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.0, hspace=0.0)
            axes = np.empty((4, 4), dtype=object)
            axes_log = np.empty((4, 4), dtype=object)
            bins = [10 ** 8, 10 ** 9, 10 ** 9.5, 10 ** 10, np.inf]
            for i in range(4):
                for j in range(4):
                    axes[i, j] = fig.add_subplot(gs[i, j])
                    axes_log[i, j] = fig_log.add_subplot(gs[i, j])

                    # Remove axis labels and ticks
                    axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                           labeltop=False, labelbottom=False)
                    axes[i, j].tick_params(axis='y', left=False, right=False,
                                           labelleft=False, labelright=False)

                    # Remove axis labels and ticks
                    axes_log[i, j].tick_params(axis='x', top=False,
                                               bottom=False,
                                               labeltop=False,
                                               labelbottom=False)
                    axes_log[i, j].tick_params(axis='y', left=False,
                                               right=False,
                                               labelleft=False,
                                               labelright=False)

            for j in range(4):
                okinds = np.logical_and(mass >= bins[j], mass < bins[j + 1])
                j_imgs = imgs[okinds, :, :]
                j_mass = mass[okinds]
                j_lumin = lumins[okinds]
                j_sd = sd[okinds]
                j_hlrs = hlrs_pix[okinds]
                done_inds = set()
                if j_sd.size != 0:
                    sdbins = np.linspace(np.min(j_sd), np.max(j_sd), 5)
                    sdbins[-1] = np.inf
                else:
                    rbins = [0, 1, 2, 3, 4]
                for i in range(4):
                    okinds = np.logical_and(j_sd >= sdbins[i],
                                            j_sd < sdbins[i + 1])
                    this_imgs = j_imgs[okinds, :, :]
                    this_mass = j_mass[okinds]
                    this_lumin = j_lumin[okinds]
                    this_sd = j_sd[okinds]
                    this_hlrs = j_hlrs[okinds]

                    try:
                        ind = np.random.choice(this_mass.size)
                    except ValueError:
                        ind = -1

                    if ind > -1:
                        size = this_imgs.shape[-1]
                        plt_img = this_imgs[ind, 
                                  int(0.3 * size):-int(0.3 * size), 
                                  int(0.3 * size):-int(0.3 * size)]
                        extent = [0, plt_img.shape[0] * csoft, 
                                  0, plt_img.shape[1] * csoft]
                        axes[i, j].imshow(plt_img,
                                          cmap=cmr.neutral_r, norm=norm, 
                                          extent=extent)

                        logimg = np.zeros_like(this_imgs[ind, :, :])
                        logimg[this_imgs[ind, :, :] > 0] = np.log10(
                            this_imgs[ind, :, :][this_imgs[ind, :, :] > 0])

                        axes_log[i, j].imshow(
                            logimg[int(0.3 * size):-int(0.3 * size),
                            int(0.3 * size):-int(0.3 * size)],
                            cmap=cmr.neutral, norm=norm_log, extent=extent)

                        string = r"$\log_{10}\left(M_\star/M_\odot\right) =$ %.2f" % np.log10(
                            this_mass[ind]) + "\n" \
                                 + r"$\log_{10}\left(L_{" + f.split(".")[
                                     -1] + r"} / [\mathrm{erg} / \mathrm{s} / " \
                                           r"\mathrm{Hz}]\right) =$ %.2f" % np.log10(
                            this_lumin[ind]) + "\n" \
                                 + r"$\log_{10}\left(S_{R<R_{1/2}," + \
                                 f.split(".")[
                                     -1] + r"} / [\mathrm{erg} / \mathrm{s} / " \
                                           r"\mathrm{Hz} / \mathrm{pkpc}^2]\right) =$ %.2f" % np.log10(
                            this_sd[
                                ind]) + " \n$R_{1/2} / [\mathrm{pkpc}] =$ %.2f" % \
                                 this_hlrs[ind]

                        axes[i, j].text(0.05, 0.95, string,
                                        transform=axes[i, j].transAxes,
                                        verticalalignment="top",
                                        horizontalalignment='left', 
                                        fontsize=SMALL_SIZE / 4,
                                        color="k")
                        axes_log[i, j].text(0.05, 0.95, string,
                                            transform=axes[i, j].transAxes,
                                            verticalalignment="top",
                                            horizontalalignment='left',
                                            fontsize=SMALL_SIZE / 4,
                                            color="w")
                    else:
                        axes[i, j].imshow(np.zeros_like(imgs[0, :, :]),
                                          cmap=cmr.neutral_r, norm=norm)
                        axes_log[i, j].imshow(np.zeros_like(imgs[0, :, :]),
                                              cmap=cmr.neutral,
                                              norm=norm_log)

            axes[3, 3].plot([0.65, 0.95], [0.05, 0.05], lw=0.5, color='k',
                    clip_on=False,
                    transform=axes[3, 3].transAxes)

            axes[3, 3].plot([0.65, 0.65], [0.025, 0.075], lw=0.5, color='k',
                    clip_on=False,
                    transform=axes[3, 3].transAxes)
            axes[3, 3].plot([0.95, 0.95], [0.025, 0.075], lw=0.5, color='k',
                    clip_on=False,
                    transform=axes[3, 3].transAxes)

            axis_to_data = axes[3, 3].transAxes \
                           + axes[3, 3].transData.inverted()
            left = axis_to_data.transform((0.65, 0.075))
            right = axis_to_data.transform((0.95, 0.075))
            print(right, left)
            dist = extent[1] * (right[0] - left[0])

            axes[3, 3].text(0.8, 0.08, "%.1f pkpc" % dist,
                    transform=axes[3, 3].transAxes, verticalalignment="top",
                    horizontalalignment='center', fontsize=SMALL_SIZE / 4,
                            color="k")
            axes_log[3, 3].text(0.8, 0.08, "%.1f pkpc" % dist,
                    transform=axes[3, 3].transAxes, verticalalignment="top",
                    horizontalalignment='center', fontsize=SMALL_SIZE / 4,
                                color="k")

            fig.savefig(
                'plots/Image_grids/ImgGrid_' + f + '_' + str(z) + '_' + reg
                + '_' + snap + '_' + orientation + '_' + Type
                + "_" + extinction + "".replace(".", "p") + ".pdf",
                bbox_inches='tight', dpi=fig.dpi)
            fig_log.savefig(
                'plots/Image_grids/LogImgGrid_' + f + '_' + str(z) + '_' + reg
                + '_' + snap + '_' + orientation + '_' + Type

                + "_" + extinction + "".replace(".", "p") + ".pdf",
                bbox_inches='tight', dpi=fig_log.dpi)
            plt.close(fig)
