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
import utilities as util
import h5py

sns.set_context("paper")
sns.set_style('whitegrid')


def get_data(ii, tag, inp='FLARES'):
    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num = '0' + num

        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"EAGLE_{inp}_sp_info.hdf5"

    with h5py.File(sim, 'r') as hf:
        S_len = np.array(hf[tag + '/Galaxy'].get('S_Length'),
                         dtype=np.int64)
        G_len = np.array(hf[tag + '/Galaxy'].get('G_Length'),
                         dtype=np.int64)
        cops = np.array(hf[tag + '/Galaxy'].get("COP"),
                        dtype=np.float64)
        S_mass = np.array(hf[tag + '/Particle'].get('S_Mass'),
                          dtype=np.float64) * 10 ** 10
        G_mass = np.array(hf[tag + '/Particle'].get('G_Mass'),
                          dtype=np.float64) * 10 ** 10
        G_Z = np.array(hf[tag + '/Particle'].get('G_Z_smooth'),
                       dtype=np.float64)
        G_coords = np.array(hf[tag + '/Particle'].get('G_Coordinates'),
                            dtype=np.float64)

    begin = np.zeros(len(S_len), dtype=np.int64)
    end = np.zeros(len(S_len), dtype=np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype=np.int64)
    gend = np.zeros(len(G_len), dtype=np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)

    return G_Z, S_len, G_len, G_coords, S_mass, G_mass, cops, begin, end, gbegin, gend


regions = []
for reg in range(0, 40):
    regions.append(reg)

snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
         '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:
        reg_snaps.append((reg, snap))

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

# Define dictionaries for results
mass_dict = {}
hdr_dict = {}

# Set mass limit
masslim = 100

for reg, tag in reg_snaps:

    print(reg, tag)

    z_str = tag.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hdr_dict.setdefault(tag, [])
    mass_dict.setdefault(tag, [])
    try:
        data = get_data(reg, tag, inp='FLARES')
    except TypeError as e:
        print(e)
        continue

    G_Z, S_len, G_len, G_coords, S_mass, G_mass, cops, \
    begin, end, gbegin, gend = data

    # Convert coordinates to physical
    G_coords = G_coords / (1 + z) * 1e3
    cops = cops / (1 + z) * 1e3

    for jj in range(len(begin)):

        if S_len[jj] < masslim:
            continue

        b, e = begin[jj], end[jj]

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        gasMetallicities = G_Z[gbegin[jj]: gend[jj]]
        gasMasses = G_mass[gbegin[jj]: gend[jj]]
        this_pos = G_coords[:, gbegin[jj]: gend[jj]].T - cops[:, jj]

        this_radii = util.calc_rad(this_pos, i=0, j=1)

        hdr_dict[tag].append(util.calc_light_mass_rad(this_radii,
                                                      gasMetallicities
                                                      * gasMasses))

        mass_dict[tag].append(np.sum(Masses))

for snap in snaps:

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hdrs = np.array(hdr_dict[snap])
    masses = np.array(mass_dict[snap])

    okinds = np.logical_and(hdrs > 0, masses > 0)

    hdrs = hdrs[okinds]
    masses = masses[okinds]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        cbar = ax.hexbin(masses, hdrs, gridsize=50, mincnt=1,
                         xscale='log', yscale='log',
                         norm=LogNorm(), linewidths=0.2,
                         cmap='viridis')
    except ValueError:
        continue

    ax.text(0.95, 0.05, f'$z={z}$',
            bbox=dict(boxstyle="round,pad=0.3", fc='w',
                      ec="k", lw=1, alpha=0.8),
            transform=ax.transAxes, horizontalalignment='right',
            fontsize=8)

    # Label axes
    ax.set_xlabel(r'$M_\star/M_\odot$')
    ax.set_ylabel('$R_{1/2, dust}/ [pkpc]$')

    fig.savefig('plots/' + str(z) + '/HalfDustRadius_'
                + str(z) + '_%d.png' % masslim,
                bbox_inches='tight')

    plt.close(fig)
