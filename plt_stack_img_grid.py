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

sns.set_context("paper")
sns.set_style('white')


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
for reg in range(0, 5):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['007_z008p000', '010_z005p000']

# reg, snap = regions[0], '010_z005p000'

np.random.seed(100)

for snap in snaps:

    reg = regions[0]

    hdf = h5py.File(
        "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, Type,
                                                    orientation),
        "r")

    f = filters[0]
    img_shape = hdf[f]["Images"].shape[-1]

    hdf.close()

    dpi = img_shape
    fig = plt.figure(figsize=(4, len(filters)), dpi=dpi)
    fig_log = plt.figure(figsize=(4, len(filters)), dpi=dpi)
    gs = gridspec.GridSpec(ncols=4, nrows=len(filters))
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((len(filters), 4), dtype=object)
    axes_log = np.empty((len(filters), 4), dtype=object)
    bins = [10 ** 8, 10 ** 9, 10 ** 9.5, 10 ** 10, np.inf]

    stacks = {}
    num_stacked = {}

    for f in filters:

        stacks[f] = {}
        num_stacked[f] = {}
        for b in bins[:-1]:
            stacks[f][b] = np.zeros((img_shape, img_shape))
            num_stacked[f][b] = 0

        for reg in regions:

            print(snap, f, reg)

            reg = regions[0]

            hdf = h5py.File(
                "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                            Type,
                                                            orientation),
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
    for f in filters:
        for b in bins[:-1]:
            stacks[f][b] = stacks[f][b] / num_stacked[f][b]
            all_imgs.append(stacks[f][b])
    all_imgs = np.array(all_imgs)
    print(all_imgs.shape)
    norm = cm.Normalize(vmin=0,
                        vmax=np.percentile(all_imgs[all_imgs > 0], 99.99),
                        clip=True)
    norm_log = cm.LogNorm(vmin=0,
                          vmax=np.percentile(all_imgs[all_imgs > 0], 99),
                          clip=True)

    for i in range(len(filters)):
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

    for j, b in enumerate(bins[:-1]):
        for i, f in enumerate(filters):

            size = stacks[f][b].shape[0]
            axes[i, j].imshow(stacks[f][b][int(0.3 * size):-int(0.3 * size),
                              int(0.3 * size):-int(0.3 * size)],
                              cmap=cmr.neutral_r, norm=norm)

            axes_log[i, j].imshow(
                stacks[f][b][int(0.3 * size):-int(0.3 * size),
                int(0.3 * size):-int(0.3 * size)],
                cmap=cmr.neutral, norm=norm_log)

    fig.savefig(
        'plots/Image_grids/StackImgGrid_' + reg
        + '_' + snap + '_' + orientation + '_' + Type
        + "_" + extinction + "".replace(".", "p") + ".png",
        bbox_inches='tight', dpi=fig.dpi)
    fig_log.savefig(
        'plots/Image_grids/StackLogImgGrid_' + reg
        + '_' + snap + '_' + orientation + '_' + Type
        + "_" + extinction + "".replace(".", "p") + ".png",
        bbox_inches='tight', dpi=fig_log.dpi)
    plt.close(fig)
