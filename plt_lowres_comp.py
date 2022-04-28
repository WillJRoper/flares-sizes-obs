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


def lowres_comp(f, regions, snap, weight_norm, orientation, extinction):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Define type
    Type = "Total"

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    lowres_hlrs = []
    normres_hlrs = []
    w = []

    for reg in regions:
        hdf_gauss = h5py.File(
            "data/flares_sizes_lowres_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                    Type,
                                                                    orientation,
                                                                    f.split(
                                                                        ".")[
                                                                        -1]),
            "r")

        grp_num_gauss = hdf_gauss[f]["GroupNumber"][...]
        subgrp_num_gauss = hdf_gauss[f]["SubGroupNumber"][...]
        hlr_gauss = hdf_gauss[f]["HLR_Pixel_0.5"][...]

        hdf_gauss.close()

        hdf_sph = h5py.File(
            "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                               Type,
                                                               orientation,
                                                               f.split(
                                                                   ".")[
                                                                   -1]),
            "r")

        grp_num_sph = hdf_sph[f]["GroupNumber"][...]
        subgrp_num_sph = hdf_sph[f]["SubGroupNumber"][...]
        hlr_sph = hdf_sph[f]["HLR_Pixel_0.5"][...]

        hdf_sph.close()
        if subgrp_num_gauss.size == subgrp_num_sph.size:

            w.extend(np.full_like(hlr_gauss, weights[int(reg)]))
            lowres_hlrs.extend(hlr_gauss)
            normres_hlrs.extend(hlr_sph)

        else:

            for (sph_ind, grp), subgrp in zip(enumerate(grp_num_sph),
                                              subgrp_num_sph):

                gauss_ind = np.where(
                    np.logical_and(grp_num_gauss == grp,
                                   subgrp_num_gauss == subgrp))[0]

                if gauss_ind.size == 0:
                    continue

                w.append(weights[int(reg)])
                lowres_hlrs.append(hlr_gauss[gauss_ind])
                normres_hlrs.append(hlr_sph[sph_ind])

        print(reg, len(w))

    lowres_hlrs = np.array(lowres_hlrs)
    normres_hlrs = np.array(normres_hlrs)
    w = np.array(w)

    okinds = np.logical_and(lowres_hlrs > 0,
                            np.logical_and(normres_hlrs > 0,
                                           np.logical_and(sph_lumins > 0,
                                                          gauss_lumins > 0)))
    lowres_hlrs = lowres_hlrs[okinds]
    normres_hlrs = normres_hlrs[okinds]
    w = w[okinds]

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(lowres_hlrs, normres_hlrs,
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
    ax.set_xlabel(r'$R_{s\times2}/ [pkpc]$')
    ax.set_ylabel('$R_{s}/ [pkpc]$')

    plt.axis('scaled')

    ax.set_xlim(10 ** -1.1, 10 ** 1.3)
    ax.set_ylim(10 ** -1.1, 10 ** 1.3)
    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.1)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("viridis"),
                                    norm=weight_norm, orientation="horizontal")
    cb1.set_label("$\sum w_{i}$")
    cb1.ax.xaxis.set_label_position('top')
    cb1.ax.xaxis.set_ticks_position('top')

    fig.savefig(
        'plots/' + str(z) + '/ResolutionComp_HLR_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)
