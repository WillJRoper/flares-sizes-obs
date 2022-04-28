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
        bin_wid = 10**0.5
        bins = np.arange(8, 12.5, np.log10(bin_wid))
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
            serr_hlrs[ind] = np.std(hlr_dict[b]) / np.sqrt(len(hlr_dict[b]))

        # Define list to store profiles
        stack_hlrs = np.zeros(len(bins[:-1]))
        stack_scale_lengths = np.zeros(len(bins[:-1]))
        stack_sl_errs = np.zeros(len(bins[:-1]))

        for f in filters:
            for j, b in enumerate(bins[:-1]):

                if not len(hlr_dict[b]) > 0:
                    continue

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

                tot_lum = np.nansum(plt_img)
                popt, pcov = curve_fit(exp_fit, xs, ys,
                                       p0=(tot_lum * 0.2, 1))

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
        ax.errorbar(bin_cents, mean_hlrs, yerr=serr_hlrs, xerr=bin_wid,
                    marker=".", capsize=5, color="k", linestyle="-",
                    label="$R_{\mathrm{pix}}$")
        ax.errorbar(bin_cents, stack_scale_lengths, yerr=stack_sl_errs,
                    xerr=bin_wid, capsize=5, marker="s", linestyle="--",
                    label="$R_{\mathrm{exp}}$")

        ax.set_ylabel("$R / [\mathrm{pkpc}]$")
        ax.set_xlabel("$M_\star / M_\odot$")

        ax.legend(loc="best")

        fig.savefig(
            'plots/' + str(z) + '/ScaleLengthComp_' + f + '_' + snap
            + '_' + orientation + '_' + Type
            + "_" + extinction + "".replace(".", "p") + ".pdf",
            bbox_inches='tight')
        plt.close(fig)
