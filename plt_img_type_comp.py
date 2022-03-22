#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import cmasher as cmr
import h5py
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

mpl.use('Agg')
warnings.filterwarnings('ignore')
from flare import plt as flareplt

# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

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


def img_size_comp(f, regions, snap, weight_norm, orientation, Type,
                  extinction):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    gauss_hlrs = []
    sph_hlrs = []
    gauss_lumins = []
    sph_lumins = []
    w = []
    gauss_imgs = []
    sph_imgs = []

    for reg in regions:
        hdf_gauss = h5py.File(
            "data/flares_sizes_gaussian_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                    Type,
                                                                    orientation,
                                                                    f.split(
                                                                        ".")[
                                                                        -1]),
            "r")

        grp_num_gauss = hdf_gauss[f]["GroupNumber"][...]
        subgrp_num_gauss = hdf_gauss[f]["SubGroupNumber"][...]
        hlr_gauss = hdf_gauss[f]["HLR_Pixel_0.5"][...]
        lumin_gauss = hdf_gauss[f]["Image_Luminosity"][...]
        gauss_img = hdf_gauss[f]["Images"][...]

        hdf_gauss.close()

        hdf_sph = h5py.File(
            "data/flares_sizes_all_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                               Type,
                                                               orientation,
                                                               f.split(
                                                                   ".")[
                                                                   -1]),
            "r")

        grp_num_sph = hdf_sph[f]["GroupNumber"][...]
        subgrp_num_sph = hdf_sph[f]["SubGroupNumber"][...]
        hlr_sph = hdf_sph[f]["HLR_Pixel_0.5"][...]
        lumin_sph = hdf_sph[f]["Image_Luminosity"][...]
        sph_img = hdf_sph[f]["Images"][...]

        hdf_sph.close()
        if subgrp_num_gauss.size == subgrp_num_sph.size:

            w.extend(np.full_like(hlr_gauss, weights[int(reg)]))
            gauss_hlrs.extend(hlr_gauss)
            sph_hlrs.extend(hlr_sph)
            gauss_lumins.extend(lumin_gauss)
            sph_lumins.extend(lumin_sph)
            gauss_imgs.extend(gauss_img)
            sph_imgs.extend(sph_img)

        else:

            for (sph_ind, grp), subgrp in zip(enumerate(grp_num_sph),
                                              subgrp_num_sph):

                gauss_ind = np.where(
                    np.logical_and(grp_num_gauss == grp,
                                   subgrp_num_gauss == subgrp))[0]

                if gauss_ind.size == 0:
                    continue

                w.append(weights[int(reg)])
                gauss_hlrs.append(hlr_gauss[gauss_ind])
                sph_hlrs.append(hlr_sph[sph_ind])
                gauss_lumins.append(lumin_gauss[gauss_ind])
                sph_lumins.append(lumin_sph[sph_ind])

        print(reg, len(w))

    gauss_hlrs = np.array(gauss_hlrs)
    sph_hlrs = np.array(sph_hlrs)
    gauss_lumins = np.array(gauss_lumins)
    sph_lumins = np.array(sph_lumins)
    w = np.array(w)

    okinds = np.logical_and(gauss_hlrs > 0,
                            np.logical_and(sph_hlrs > 0,
                                           np.logical_and(sph_lumins > 0,
                                                          gauss_lumins > 0)))
    gauss_hlrs = gauss_hlrs[okinds]
    sph_hlrs = sph_hlrs[okinds]
    gauss_lumins = gauss_lumins[okinds]
    sph_lumins = sph_lumins[okinds]
    w = w[okinds]

    gimg = np.nansum(gauss_imgs, axis=0)
    simg = np.nansum(sph_imgs, axis=0)
    resi = (np.log10(gimg) - np.log10(simg))

    dpi = gimg.shape[0] * 2
    fig = plt.figure(figsize=(3 * 3.5, 3.5), dpi=dpi)
    gs = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=[10, 10, 10, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    for ax in [ax1, ax2, ax3]:
        # Remove axis labels and ticks
        ax.tick_params(axis='x', top=False, bottom=False,
                       labeltop=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, right=False,
                       labelleft=False, labelright=False)
        ax.grid(False)

    log_norm = cm.LogNorm(vmin=np.percentile(gimg, 16),
                          vmax=np.percentile(simg, 99), clip=True)
    diverg_norm = cm.TwoSlopeNorm(vmin=-np.nanmax((np.abs(resi))),
                                  vcenter=0.,
                                  vmax=np.nanmax((np.abs(resi))))

    ax1.imshow(gimg, cmap=cmr.neutral,
               norm=log_norm)
    ax2.imshow(simg, cmap=cmr.neutral,
               norm=log_norm)
    im = ax3.imshow(resi, cmap=cmr.iceburn,
                    norm=diverg_norm)

    ax1.text(0.05, 0.875,
             "Gaussian",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc='grey',
                       ec="w", lw=1, alpha=0.7),
             transform=ax1.transAxes,
             horizontalalignment='left', color="w")

    ax2.text(0.05, 0.875,
             "Spline",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc='grey',
                       ec="w", lw=1, alpha=0.7),
             transform=ax2.transAxes,
             horizontalalignment='left', color="w")

    ax3.text(0.05, 0.875,
             "Gaussian - Spline",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc='grey',
                       ec="w", lw=1, alpha=0.7),
             transform=ax3.transAxes,
             horizontalalignment='left', color="w")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("$\Delta \log_{10}(L /$ [erg $/$ s $/$ Hz])")

    fig.savefig(
        'plots/' + str(z) + '/ComparisonImageCreation_Residual_'
        + f + '_' + str(z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(gauss_hlrs, sph_hlrs,
                         C=w, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3))
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** -1.1, 10 ** 1.3], [10 ** -1.1, 10 ** 1.3],
            color='k', linestyle="--")

    # Label axes
    ax.set_xlabel('$R_{\mathrm{gauss}}/ [pkpc]$')
    ax.set_ylabel('$R_{\mathrm{spline}}/ [pkpc]$')

    plt.axis('scaled')

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)
    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.1)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("viridis"), norm=weight_norm, orientation="horizontal")
    cb1.set_label("$\sum w_{i}$")
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')

    fig.savefig(
        'plots/' + str(z) + '/ComparisonImageCreation_HLR_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(gauss_lumins, sph_lumins,
                         C=w, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis')
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** 27., 10 ** 31.], [10 ** 27., 10 ** 31.],
            color='k', linestyle="--")

    # Label axes
    ax.set_xlabel('$L_{\mathrm{gauss}}/$ [erg $/$ s $/$ Hz]')
    ax.set_ylabel('$L_{\mathrm{spline}}/$ [erg $/$ s $/$ Hz]')

    plt.axis('scaled')

    ax.set_xlim(10 ** 27., 10 ** 31.)
    ax.set_ylim(10 ** 27., 10 ** 31.)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.1)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("viridis"), norm=weight_norm, orientation="horizontal")
    cb1.set_label("$\sum w_{i}$")
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    fig.savefig(
        'plots/' + str(z) + '/ComparisonImageCreation_Lumin_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')
