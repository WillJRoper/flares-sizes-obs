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

filter_path = "/cosma7/data/dp004/dc-wilk2/flare/data/filters/"


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
orientation = "sim"

# Define luminosity and dust model types
extinction = 'default'

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV', 'FAKE.TH.V')

csoft = 0.001802390 / (0.6777) * 1e3

sedint_dict = {}
sedtot_dict = {}
sedlam_dict = {}
imgtot_dict = {}
imgint_dict = {}
mass_dict = {}

regions = []
for reg in range(35, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['005_z010p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000']

lim = 10

np.random.seed(100)

for reg in regions:
    for snap in snaps:

        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Total",
                                                        orientation),
            "r")

        for f in filters:
            sedint_dict[f] = hdf[f]["SED_intrinsic"][...]
            sedtot_dict[f] = hdf[f]["SED_total"][...]
            sedlam_dict[f] = hdf[f]["SED_lambda"][...]
            imgtot_dict[f] = hdf[f]["Images"][...]
            mass_dict[f] = hdf[f]["Mass"][...]

        hdf.close()

        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Intrinsic",
                                                        orientation),
            "r")

        for f in filters:
            imgint_dict[f] = hdf[f]["Images"][...]

        hdf.close()

        for f in filters:

            print("Plotting for:")
            print("Region = ", reg)
            print("Snapshot = ", snap)
            print("Orientation =", orientation)
            print("Filter =", f)

            l, t = np.loadtxt(filter_path + '/' + '/'.join(f.split('.')) + '.txt', skiprows=1).T

            print(np.min(l[t > 0]), np.max(l[t > 0]))

            legend_elements = []

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            sedint = np.array(sedint_dict[f])
            sedtot = np.array(sedtot_dict[f])
            sedlam = np.array(sedlam_dict[f])
            imgtot = np.array(imgtot_dict[f])
            imgint = np.array(imgint_dict[f])
            masses = np.array(mass_dict[f])

            fig = plt.figure(figsize=(8, 3))
            ax = fig.add_subplot(111)
            ax.loglog()

            ax.axvspan(np.min(l[t > 0]), np.max(l[t > 0]), alpha=0.5,
                       color='cyan')

            i = 0
            done = set()
            while i < lim:
                ind = np.random.choice(len(masses))
                j = 0
                while ind in done:
                    ind = np.random.choice(len(masses))
                    j += 1
                    if j > lim:
                        i = lim + 1
                        break
                ax.plot(sedlam[ind, :], sedtot[ind, :], color="r", alpha=0.05)
                ax.plot(sedlam[ind, :], sedint[ind, :], color="g", alpha=0.05)
                done.update({ind})

            max_ind = np.argmax(masses)
            ax.plot(sedlam[max_ind, :], sedtot[max_ind, :], color="r")
            ax.plot(sedlam[max_ind, :], sedint[max_ind, :], color="g")

            fig.savefig(
                'plots/SED/SED' + f + '_' + str(z) + '_' + reg
                + '_' + snap + '_' + orientation + "_"
                + extinction + "".replace(".", "p") + ".png",
                bbox_inches='tight', dpi=fig.dpi)
            plt.close(fig)
