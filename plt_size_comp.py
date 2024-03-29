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
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from flare.photom import lum_to_M, M_to_lum
import flare.photom as photconv
import h5py
import sys
import pandas as pd
import utilities as util

sns.set_context("paper")
sns.set_style('whitegrid')


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
Type = "Intrinsic"
extinction = 'default'

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

# Define filter
filters = ('FAKE.TH.FUV', )

csoft = 0.001802390 / (0.6777) * 1e3

nlim = 800

hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
weight_dict = {}

lumin_bins = np.logspace(np.log10(M_to_lum(-16)), np.log10(M_to_lum(-24)), 20)
M_bins = np.linspace(-24, -16, 20)

lumin_bin_wid = lumin_bins[1] - lumin_bins[0]
M_bin_wid = M_bins[1] - M_bins[0]

lumin_bin_cents = lumin_bins[1:] - (lumin_bin_wid / 2)
M_bin_cents = M_bins[1:] - (M_bin_wid / 2)

# Load weights
df = pd.read_csv('../weight_files/weights_grid.txt')
weights = np.array(df['weights'])

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

for num, (reg, snap) in enumerate(reg_snaps):

    hdf = h5py.File("data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
    try:
        type_group = hdf[Type]
        orientation_group = type_group[orientation]
    except KeyError as e:
        print(reg, snap, e)
        continue

    hlr_dict.setdefault(snap, {})
    hlr_app_dict.setdefault(snap, {})
    hlr_pix_dict.setdefault(snap, {})
    lumin_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hlr_app_dict[snap].setdefault(f, [])
        hlr_pix_dict[snap].setdefault(f, [])
        lumin_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])

        masses = orientation_group[f]["Mass"][...]
        okinds = orientation_group[f]["nStar"][...] > nlim

        print(reg, snap, f, masses[okinds].size, num)

        hlr_dict[snap][f].extend(orientation_group[f]["HLR_0.5"][...][okinds])
        hlr_app_dict[snap][f].extend(
            orientation_group[f]["HLR_Aperture_0.5"][...][okinds])
        hlr_pix_dict[snap][f].extend(
            orientation_group[f]["HLR_Pixel_0.5"][...][okinds])
        lumin_dict[snap][f].extend(
            orientation_group[f]["Luminosity"][...][okinds])
        mass_dict[snap][f].extend(masses[okinds])
        weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))

    hdf.close()

for f in filters:

    fit_lumins = np.logspace(np.log10(M_to_lum(-21.6)),
                             np.log10(M_to_lum(-18)),
                             1000)

    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)

    legend_elements = []

    for snap in snaps:

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])
        if snap in hlr_dict:
            hlrs = np.array(hlr_dict[snap][f])
            hlrs_app = np.array(hlr_app_dict[snap][f])
            hlrs_pix = np.array(hlr_pix_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])
            mass = np.array(mass_dict[snap][f])
        else:
            continue

        okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                np.logical_and(lumins > M_to_lum(-12),
                                               lumins < 10 ** 50))
        lumins = lumins[okinds]
        hlrs = hlrs[okinds]
        hlrs_app = hlrs_app[okinds]
        hlrs_pix = hlrs_pix[okinds]
        mass = mass[okinds]

        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        try:
            cbar = ax1.hexbin(hlrs, hlrs_app, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                              xscale='log', yscale='log',
                              norm=LogNorm(), linewidths=0.2,
                              cmap='viridis')
            cbar = ax2.hexbin(hlrs, hlrs_pix, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                              xscale='log', yscale='log',
                              norm=LogNorm(), linewidths=0.2,
                              cmap='viridis')
            cbar = ax3.hexbin(hlrs_app, hlrs_pix, gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                              xscale='log', yscale='log',
                              norm=LogNorm(), linewidths=0.2,
                              cmap='viridis')
        except ValueError as e:
            print(e)
            continue

        for ax in [ax1, ax2, ax3]:
            axis_to_data = ax.transAxes + ax.transData.inverted()
            left = axis_to_data.transform((0, 0))
            right = axis_to_data.transform((1, 1))
            ax.plot((left[0], right[0]), (left[1], right[1]),
                    color="k", linestyle="--")

        # Label axes
        ax2.set_xlabel('$R_{1/2, \mathrm{part}}/ [pkpc]$')
        ax3.set_xlabel('$R_{1/2, \mathrm{app}}/ [pkpc]$')
        ax1.set_ylabel('$R_{1/2, \mathrm{app}}/ [pkpc]$')
        ax2.set_ylabel('$R_{1/2, \mathrm{pixel}}/ [pkpc]$')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax3.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        fig.savefig(
            'plots/' + str(z) + '/ComparisonHalfLightRadius_' + f + '_' + str(
                z) + '_'
            + orientation + '_' + Type + "_" + extinction + "_"
            + '%d.png' % nlim,
            bbox_inches='tight')

        plt.close(fig)
