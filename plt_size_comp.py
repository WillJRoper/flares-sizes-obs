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
from FLARE.photom import lum_to_M, M_to_lum
import FLARE.photom as photconv
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
Type = sys.argv[2]
extinction = 'default'

if sys.argv[3] == "All":
    snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']
else:
    snaps = sys.argv[3]

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

csoft = 0.001802390 / (0.6777) * 1e3

masslim = 10 ** float(sys.argv[4])

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

for reg, snap in reg_snaps:

    hdf = h5py.File("data/flares_sizes_{}_{}.hdf5".format(reg, snap), "r")
    type_group = hdf[Type]
    orientation_group = type_group[orientation]

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
        okinds = masses > masslim

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

        hlrs = np.array(hlr_dict[snap][f])
        hlrs_app = np.array(hlr_app_dict[snap][f])
        hlrs_pix = np.array(hlr_pix_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        mass = np.array(mass_dict[snap][f])

        okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                np.logical_and(lumins > M_to_lum(-12),
                                               lumins < 10 ** 50))
        lumins = lumins[okinds]
        hlrs = hlrs[okinds]
        hlrs_app = hlrs_app[okinds]
        hlrs_pix = hlrs_pix[okinds]
        mass = mass[okinds]

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            cbar = ax1.hexbin(hlrs, hlrs_app, gridsize=50, mincnt=1,
                              C=lumins,
                              reduce_C_function=np.mean,
                              xscale='log', yscale='log',
                              norm=LogNorm(), linewidths=0.2,
                              cmap='viridis')
            cbar = ax1.hexbin(hlrs, hlrs_pix, gridsize=50, mincnt=1,
                              C=lumins,
                              reduce_C_function=np.mean,
                              xscale='log', yscale='log',
                              norm=LogNorm(), linewidths=0.2,
                              cmap='viridis')
        except ValueError as e:
            print(e)
            continue

        ax.text(0.8, 0.1, f'$z={z}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w',
                          ec="k", lw=1, alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right',
                fontsize=8)

        # Label axes
        ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        ax.legend(loc="bottom right")

        fig.savefig(
            'plots/' + str(z) + '/HalfLightRadius_' + f + '_' + str(
                z) + '_'
            + orientation + '_' + Type + "_" + extinction + "_"
            + '%.1f.png' % np.log10(masslim),
            bbox_inches='tight')

        plt.close(fig)
