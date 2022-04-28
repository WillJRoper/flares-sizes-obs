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


def noBC_comp(f, regions, snap, weight_norm, orientation,
                  extinction):
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    
    # Define type
    Type = "Total"

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    noBC_hlrs = []
    withBC_hlrs = []
    w = []

    for reg in regions:
        hdf_noBC = h5py.File(
            "data/flares_sizes_noBC_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                    Type,
                                                                    orientation,
                                                                    f.split(
                                                                        ".")[
                                                                        -1]),
            "r")

        grp_num_noBC = hdf_noBC[f]["GroupNumber"][...]
        subgrp_num_noBC = hdf_noBC[f]["SubGroupNumber"][...]
        hlr_noBC = hdf_noBC[f]["HLR_Pixel_0.5"][...]

        hdf_noBC.close()

        hdf_withBC = h5py.File(
            "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg,
                                                                         snap,
                                                                         Type,
                                                                         orientation,
                                                                         f.split(
                                                                             ".")[
                                                                             -1]),
            "r")

        grp_num_withBC = hdf_withBC[f]["GroupNumber"][...]
        subgrp_num_withBC = hdf_withBC[f]["SubGroupNumber"][...]
        hlr_withBC = hdf_withBC[f]["HLR_Pixel_0.5"][...]

        hdf_withBC.close()
        if subgrp_num_noBC.size == subgrp_num_withBC.size:

            w.extend(np.full_like(hlr_noBC, weights[int(reg)]))
            noBC_hlrs.extend(hlr_noBC)
            withBC_hlrs.extend(hlr_withBC)

        else:

            for (withBC_ind, grp), subgrp in zip(enumerate(grp_num_withBC),
                                              subgrp_num_withBC):

                noBC_ind = np.where(
                    np.logical_and(grp_num_noBC == grp,
                                   subgrp_num_noBC == subgrp))[0]

                if noBC_ind.size == 0:
                    continue

                w.append(weights[int(reg)])
                noBC_hlrs.append(hlr_noBC[noBC_ind])
                withBC_hlrs.append(hlr_withBC[withBC_ind])

        print(reg, len(w))

    noBC_hlrs = np.array(noBC_hlrs)
    withBC_hlrs = np.array(withBC_hlrs)
    w = np.array(w)

    okinds = np.logical_and(noBC_hlrs > 0, withBC_hlrs > 0)
    noBC_hlrs = noBC_hlrs[okinds]
    withBC_hlrs = withBC_hlrs[okinds]
    w = w[okinds]

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(withBC_hlrs, noBC_hlrs,
                         C=w, gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=(-1.1, 1.3, -1.1, 1.3))
    except ValueError as e:
        print(e)
        return

    ax.plot([10 ** -1.1, 10 ** 1.3], [10 ** -1.1, 10 ** 1.3],
            color='k', linestyle="--")

    print(np.max(withBC_hlrs/noBC_hlrs))

    # Label axes
    ax.set_xlabel(r'$R_{\mathrm{withBC}}/ [pkpc]$')
    ax.set_ylabel('$R_{\mathrm{noBC}}/ [pkpc]$')

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
        'plots/' + str(z) + '/BirthCloudComp_HLR_' + f + '_' + str(
            z) + '_'
        + orientation + '_' + Type + "_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)

