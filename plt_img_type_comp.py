#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns

sns.set_context("paper")
sns.set_style('whitegrid')


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

        hdf_sph.close()
        print(grp_num_gauss, subgrp_num_gauss)
        print(grp_num_gauss.size, subgrp_num_gauss.size,
              grp_num_sph.size, subgrp_num_sph.size)

        for (sph_ind, grp), subgrp in zip(enumerate(grp_num_sph),
                                          subgrp_num_sph):

            print(grp, subgrp)

            gauss_ind = np.where(np.logical_and(grp_num_gauss == grp,
                                                subgrp_num_gauss == subgrp))[0]

            if gauss_ind.size == 0:
                continue

            w.append(weights[int(reg)])
            gauss_hlrs.append(hlr_gauss[gauss_ind])
            sph_hlrs.append(hlr_sph[sph_ind])
            gauss_lumins.append(lumin_gauss[gauss_ind])
            sph_lumins.append(lumin_sph[sph_ind])

        print(reg, len(w))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(gauss_hlrs, sph_hlrs,
                         C=w, gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3))
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** -1.1, 10 ** 1.3], [10 ** -1.1, 10 ** 1.3],
            color='k', linestyle="--")

    # Label axes
    ax.set_xlabel('$R_{1/2, \mathrm{gauss}}/ [pkpc]$')
    ax.set_ylabel('$R_{1/2, \mathrm{spline}}/ [pkpc]$')

    plt.axis('scaled')

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)

    fig.savefig(
        'plots/' + str(z) + '/ComparisonImageCreation_HLR_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".png",
        bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(gauss_lumins, sph_lumins,
                         C=w, gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis')
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** 27, 10 ** 31], [10 ** 27, 10 ** 31],
            color='k', linestyle="--")

    # Label axes
    ax.set_xlabel('$L_{\mathrm{gauss}}/ [erg $/$ s $/$ Hz]"$')
    ax.set_ylabel('$L_{\mathrm{spline}}/ [erg $/$ s $/$ Hz]"$')

    plt.axis('scaled')

    ax.set_xlim(10 ** 27, 10 ** 31)
    ax.set_ylim(10 ** 27, 10 ** 31)

    fig.savefig(
        'plots/' + str(z) + '/ComparisonImageCreation_Lumin_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".png",
        bbox_inches='tight')
