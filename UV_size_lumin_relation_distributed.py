#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from photutils import CircularAperture

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import phot_modules as phot
import utilities as util
from FLARE.photom import lum_to_M, M_to_lum
import h5py
import sys
import time

sns.set_context("paper")
sns.set_style('whitegrid')

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_up_params = {'beta': {6: 0.08, 7: 0.08,
                           8: 0.28, 9: 1.01},
                  'r_0': {6: 0.2, 7: 0.2,
                          8: 5.28, 9: 367.64}}
kawa_low_params = {'beta': {6: 0.09, 7: 0.09,
                            8: 0.78, 9: 0.27},
                   'r_0': {6: 0.15, 7: 0.15,
                           8: 0.26, 9: 0.74}}
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b


def kawa_fit_err(y, l, ro, b, ro_err, b_err, uplow="up"):
    ro_term = ro_err * (l / M_to_lum(-21)) ** b
    beta_term = b_err * ro * (l / M_to_lum(-21)) ** b \
                * np.log(l / M_to_lum(-21))

    if uplow == "up":
        return y + np.sqrt(ro_term ** 2 + beta_term ** 2)
    else:
        return y - np.sqrt(ro_term ** 2 + beta_term ** 2)


def plot_meidan_stat(xs, ys, ax, lab, color, bins=None, ls='-'):
    if bins == None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 15)
    else:
        bin = bins

    # Compute binned statistics
    y_stat, binedges, bin_ind = binned_statistic(xs, ys, statistic='median',
                                                 bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    ax.plot(bin_cents[okinds], y_stat[okinds], color=color, linestyle=ls,
            label=lab)


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
         '003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000', '011_z004p770']

reg_snaps = []
for reg in reversed(regions):

    for snap in snaps:

        reg_snaps.append((reg, snap))


ind = int(sys.argv[1])

# Set orientation
orientation = sys.argv[2]

# Define luminosity and dust model types
Type = sys.argv[3]
extinction = 'default'

reg, tag = reg_snaps[ind]
print("Computing HLRs with orientation {o}, type {t}, and extinction {e}"
      "for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

# Define radii
radii_fracs = (0.2, 0.5, 0.8)

# Define dictionaries for results
hlr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
mass_dict = {}
img_dict = {}

# Set mass limit
masslim = 100

z_str = tag.split('z')[1].split('p')
z = float(z_str[0] + '.' + z_str[1])

hlr_dict.setdefault(tag, {})
hlr_app_dict.setdefault(tag, {})
hlr_pix_dict.setdefault(tag, {})
lumin_dict.setdefault(tag, {})
mass_dict.setdefault(tag, {})
img_dict.setdefault(tag, {})

# Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
reg_dict = phot.get_lum(reg, kappa=0.0795, tag=tag, BC_fac=1,
                        IMF='Chabrier_300', bins=np.arange(-24, -16, 0.5),
                        inp='FLARES', LF=False, filters=filters, Type=Type,
                        log10t_BC=7., extinction=extinction,
                        orientation=orientation, masslim=masslim)

if z <= 2.8:
    csoft = 0.000474390 / 0.6777 * 1e3
else:
    csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

# Define width
ini_width = 60

# Compute the resolution
ini_res = ini_width / csoft
res = int(np.ceil(ini_res))

# Compute the new width
width = csoft * res

print(width, res)

single_pixel_area = csoft * csoft

# Define range and extent for the images
imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
imgextent = [-width / 2, width / 2, -width / 2, width / 2]

# Set up aperture objects
positions = [(res / 2, res / 2)]
app_radii = np.linspace(0.001, res / 4, 100)
apertures = [CircularAperture(positions, r=r) for r in app_radii]
app_radii *= csoft

poss = reg_dict["coords"] * 10 ** 3
smls = reg_dict["smls"] * 10 ** 3
masses = reg_dict["masses"]
begin = reg_dict["begin"]
end = reg_dict["end"]

for f in filters:

    hlr_dict[tag].setdefault(f, {})
    hlr_app_dict[tag].setdefault(f, {})
    hlr_pix_dict[tag].setdefault(f, {})

    for r in radii_fracs:
        hlr_dict[tag][f].setdefault(r, [])
        hlr_app_dict[tag][f].setdefault(r, [])
        hlr_pix_dict[tag][f].setdefault(r, [])

    lumin_dict[tag].setdefault(f, [])
    mass_dict[tag].setdefault(f, [])
    img_dict[tag].setdefault(f, [])

    for ind in range(len(begin)):

        b, e = begin[ind], end[ind]

        this_pos = poss[:, b: e].T
        this_lumin = reg_dict[f][b: e]
        this_smls = smls[b: e]
        this_mass = np.nansum(masses[b: e])

        if np.nansum(this_lumin) == 0:
            continue

        tot_l = np.sum(this_lumin)

        if orientation == "sim" or orientation == "face-on":

            # # Centre positions on luminosity weighted centre
            # # NOTE: This is done in 3D not in projection!
            # lumin_cent = util.lumin_weighted_centre(this_pos,
            #                                         this_lumin,
            #                                         i=0, j=1)
            # this_pos[:, (0, 1)] -= lumin_cent

            this_radii = util.calc_rad(this_pos, i=0, j=1)

            # img = util.make_soft_img(this_pos, res, 0, 1, imgrange,
            #                          this_lumin,
            #                          this_smls)

            img = np.histogram2d(this_pos[:, 0], this_pos[:, 1], bins=res, range=imgrange, weights=this_lumin)

        else:

            # # Centre positions on luminosity weighted centre
            # # NOTE: This is done in 3D not in projection!
            # lumin_cent = util.lumin_weighted_centre(this_pos,
            #                                         this_lumin,
            #                                         i=2, j=0)
            # this_pos[:, (2, 0)] -= lumin_cent

            this_radii = util.calc_rad(this_pos, i=2, j=0)

            # img = util.make_soft_img(this_pos, res, 2, 0, imgrange,
            #                          this_lumin,
            #                          this_smls)

            img = np.histogram2d(this_pos[:, 2], this_pos[:, 0], bins=res, range=imgrange, weights=this_lumin)

        for r in radii_fracs:

            hlr_app_dict[tag][f][r].append(util.get_img_hlr(img,
                                                            apertures,
                                                            app_radii, res,
                                                            csoft, r))

            hlr_pix_dict[tag][f][r].append(
                util.get_pixel_hlr(img, single_pixel_area, r))

            hlr_dict[tag][f][r].append(util.calc_light_mass_rad(this_radii,
                                                                this_lumin,
                                                                r))

        lumin_dict[tag][f].append(tot_l)
        mass_dict[tag][f].append(this_mass)
        img_dict[tag][f].append(img)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(np.log10(img), extent=imgextent)
        # ax.grid(False)
        # circle1 = plt.Circle((0, 0), 30, color='r', fill=False)
        # ax.add_artist(circle1)
        # circle1 = plt.Circle((0, 0), hlr_app_dict[tag][f][-1],
        #                      color='g', linestyle="--", fill=False)
        # ax.add_artist(circle1)
        # circle1 = plt.Circle((0, 0), hlr_dict[tag][f][-1],
        #                      color='b', linestyle="--", fill=False)
        # ax.add_artist(circle1)
        # fig.savefig("plots/gal_img_log_%.1f.png"
        #             % np.log10(np.sum(this_lumin)))
        # plt.close(fig)

try:
    hdf = h5py.File("data/flares_sizes_noSmooth_{}_{}.hdf5".format(reg, tag), "r+")
except OSError:
    hdf = h5py.File("data/flares_sizes_noSmooth_{}_{}.hdf5".format(reg, tag), "w")

try:
    type_group = hdf[Type]
except KeyError:
    type_group = hdf.create_group(Type)

try:
    orientation_group = type_group[orientation]
except KeyError:
    orientation_group = type_group.create_group(orientation)

for f in filters:

    lumins = np.array(lumin_dict[tag][f])
    mass = np.array(mass_dict[tag][f])
    imgs = np.array(img_dict[tag][f])

    print(imgs.shape)

    try:
        f_group = orientation_group[f]
    except KeyError:
        f_group = orientation_group.create_group(f)

    try:
        dset = f_group.create_dataset("Luminosity", data=lumins,
                                      dtype=lumins.dtype,
                                      shape=lumins.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
    except RuntimeError:
        del f_group["Luminosity"]
        dset = f_group.create_dataset("Luminosity", data=lumins,
                                      dtype=lumins.dtype,
                                      shape=lumins.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

    try:
        dset = f_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
    except RuntimeError:
        del f_group["Images"]
        dset = f_group.create_dataset("Images", data=imgs,
                                      dtype=imgs.dtype,
                                      shape=imgs.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

    try:
        dset = f_group.create_dataset("Mass", data=mass,
                                      dtype=mass.dtype,
                                      shape=mass.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$M_\odot$"
    except RuntimeError:
        del f_group["Mass"]
        dset = f_group.create_dataset("Mass", data=mass,
                                      dtype=mass.dtype,
                                      shape=mass.shape,
                                      compression="gzip")
        dset.attrs["units"] = "$M_\odot$"

    for r in radii_fracs:

        hlrs = np.array(hlr_dict[tag][f][r])
        hlrs_app = np.array(hlr_app_dict[tag][f][r])
        hlrs_pix = np.array(hlr_pix_dict[tag][f][r])

        try:
            dset = f_group.create_dataset("HLR_%.1f" % r, data=hlrs,
                                          dtype=hlrs.dtype,
                                          shape=hlrs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            del f_group["HLR_%.1f" % r]
            dset = f_group.create_dataset("HLR_%.1f" % r, data=hlrs,
                                          dtype=hlrs.dtype,
                                          shape=hlrs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

        try:
            dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
                                          data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            del f_group["HLR_Aperture_%.1f" % r]
            dset = f_group.create_dataset("HLR_Aperture_%.1f" % r,
                                          data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

        try:
            dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
                                          data=hlrs_pix,
                                          dtype=hlrs_pix.dtype,
                                          shape=hlrs_pix.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            del f_group["HLR_Pixel_%.1f" % r]
            dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
                                          data=hlrs_pix,
                                          dtype=hlrs_pix.dtype,
                                          shape=hlrs_pix.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
