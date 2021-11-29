#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import numpy as np
from photutils import CircularAperture

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
warnings.filterwarnings('ignore')
from scipy.stats import binned_statistic
import phot_modules as phot
import utilities as util
from flare.photom import M_to_lum
from scipy.spatial import cKDTree
import h5py
import sys
from synthobs.sed import models
import flare.filters


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
print("Computing HLRs with orientation {o}, type {t}, and extinction {e} "
      "for region {x} and snapshot {u}".format(o=orientation, t=Type,
                                               e=extinction, x=reg, u=tag))

# Define filter
# filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV', 'FAKE.TH.V')
filters = ['FAKE.TH.' + f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]

run = True

model = models.define_model(
    F'BPASSv2.2.1.binary/Chabrier_300')  # DEFINE SED GRID -

model.dust_ISM = (
    'simple', {'slope': -1.})  # Define dust curve for ISM
model.dust_BC = ('simple', {
    'slope': -1.})  # Define dust curve for birth cloud component

# --- create rest-frame luminosities
F = flare.filters.add_filters(filters, new_lam=model.lam)
model.create_Lnu_grid(
    F)  # --- create new L grid for each filter. In units of erg/s/Hz

if run:

    # Define radii
    radii_fracs = (0.2, 0.5, 0.8)

    # Define dictionaries for results
    hdr_dict = {}
    hlr_dict = {}
    hlr_app_dict = {}
    hlr_pix_dict = {}
    nosmooth_hlr_dict = {}
    nosmooth_hlr_app_dict = {}
    nosmooth_hlr_pix_dict = {}
    lumin_dict = {}
    img_lumin_dict = {}
    nosmooth_img_lumin_dict = {}
    mass_dict = {}
    gmass_dict = {}
    met_dict = {}
    gmet_dict = {}
    nstar_dict = {}
    img_dict = {}
    nosmooth_img_dict = {}
    sed_int = {}
    sed_tot = {}
    sed_lam = {}
    surf_dens = {}

    # Set mass limit
    masslim = 10 ** 8
    nlim = None

    z_str = tag.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hdr_dict.setdefault(tag, {})
    hlr_dict.setdefault(tag, {})
    hlr_app_dict.setdefault(tag, {})
    hlr_pix_dict.setdefault(tag, {})
    nosmooth_hlr_dict.setdefault(tag, {})
    nosmooth_hlr_app_dict.setdefault(tag, {})
    nosmooth_hlr_pix_dict.setdefault(tag, {})
    lumin_dict.setdefault(tag, {})
    img_lumin_dict.setdefault(tag, {})
    nosmooth_img_lumin_dict.setdefault(tag, {})
    mass_dict.setdefault(tag, {})
    gmass_dict.setdefault(tag, {})
    met_dict.setdefault(tag, {})
    gmet_dict.setdefault(tag, {})
    nstar_dict.setdefault(tag, {})
    img_dict.setdefault(tag, {})
    nosmooth_img_dict.setdefault(tag, {})
    sed_int.setdefault(tag, {})
    sed_tot.setdefault(tag, {})
    sed_lam.setdefault(tag, {})
    surf_dens.setdefault(tag, {})

    # Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25
    reg_dict = phot.get_lum(reg, kappa=0.0795, tag=tag, BC_fac=1,
                            IMF='Chabrier_300', bins=np.arange(-24, -16, 0.5),
                            inp='FLARES', LF=False, filters=filters, Type=Type,
                            log10t_BC=7., extinction=extinction,
                            orientation=orientation, masslim=masslim,
                            nlim=nlim)
    print("Got luminosities")

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

    # Define x and y positions of pixels
    X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], res),
                       np.linspace(imgrange[1][0], imgrange[1][1], res))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 2))
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()

    # Build KDTree
    tree = cKDTree(pix_pos)

    print("Pixel tree built")

    # Set up aperture objects
    positions = [(res / 2, res / 2)]
    app_radii = np.linspace(0.001, res / 4, 100)
    apertures = [CircularAperture(positions, r=r) for r in app_radii]
    app_radii *= csoft

    poss = reg_dict["coords"] * 10 ** 3
    gposs = reg_dict["gcoords"] * 10 ** 3
    smls = reg_dict["smls"] * 10 ** 3
    masses = reg_dict["masses"]
    gas_masses = reg_dict["gmasses"]
    star_Z = reg_dict["S_Z"]
    gas_Z = reg_dict["G_Z"]
    nstars = reg_dict["nstar"]
    begin = reg_dict["begin"]
    end = reg_dict["end"]
    gbegin = reg_dict["gbegin"]
    gend = reg_dict["gend"]

    for f in filters:

        hdr_dict[tag].setdefault(f, [])
        hlr_dict[tag].setdefault(f, {})
        hlr_app_dict[tag].setdefault(f, {})
        hlr_pix_dict[tag].setdefault(f, {})
        nosmooth_hlr_dict[tag].setdefault(f, {})
        nosmooth_hlr_app_dict[tag].setdefault(f, {})
        nosmooth_hlr_pix_dict[tag].setdefault(f, {})

        for r in radii_fracs:
            hlr_dict[tag][f].setdefault(r, [])
            hlr_app_dict[tag][f].setdefault(r, [])
            hlr_pix_dict[tag][f].setdefault(r, [])
            nosmooth_hlr_dict[tag][f].setdefault(r, [])
            nosmooth_hlr_app_dict[tag][f].setdefault(r, [])
            nosmooth_hlr_pix_dict[tag][f].setdefault(r, [])

        lumin_dict[tag].setdefault(f, [])
        img_lumin_dict[tag].setdefault(f, [])
        nosmooth_img_lumin_dict[tag].setdefault(f, [])
        mass_dict[tag].setdefault(f, [])
        gmass_dict[tag].setdefault(f, [])
        met_dict[tag].setdefault(f, [])
        gmet_dict[tag].setdefault(f, [])
        nstar_dict[tag].setdefault(f, [])
        img_dict[tag].setdefault(f, [])
        nosmooth_img_dict[tag].setdefault(f, [])
        sed_int[tag].setdefault(f, [])
        sed_tot[tag].setdefault(f, [])
        sed_lam[tag].setdefault(f, [])
        surf_dens[tag].setdefault(f, [])

        for ind in range(len(begin)):

            b, e = begin[ind], end[ind]
            gb, ge = gbegin[ind], gend[ind]

            this_pos = poss[:, b: e].T
            this_gpos = gposs[:, gb: ge].T
            this_lumin = reg_dict[f][b: e]
            this_smls = smls[b: e]
            this_mass = np.nansum(masses[b: e])
            this_gmass = np.nansum(gas_masses[b: e])
            this_met = star_Z[gb: ge] * masses[gb: ge]
            this_metals = gas_Z[gb: ge] * gas_masses[gb: ge]
            this_nstar = nstars[ind]
            this_age = reg_dict["S_age"][b: e]
            this_Sz = reg_dict["S_Z"][b: e]

            if np.nansum(this_lumin) == 0:
                continue

            tot_l = np.sum(this_lumin)

            if orientation == "sim" or orientation == "face-on":

                # # Centre positions on luminosity weighted centre
                # lumin_cent = util.lumin_weighted_centre(this_pos,
                #                                         this_lumin,
                #                                         i=0, j=1)
                # this_pos[:, (0, 1)] -= lumin_cent

                this_radii = util.calc_rad(this_pos, i=0, j=1)
                this_gradii = util.calc_rad(this_gpos, i=0, j=1)

                img = util.make_spline_img(this_pos, res, 0, 1, tree,
                                           this_lumin, this_smls,
                                           spline_func=util.cubic_spline,
                                           spline_cut_off=1)

                no_smooth_img = np.histogram2d(this_pos[:, 0], this_pos[:, 1],
                                               range=imgrange, bins=res)

            else:

                # # Centre positions on luminosity weighted centre
                # lumin_cent = util.lumin_weighted_centre(this_pos,
                #                                         this_lumin,
                #                                         i=2, j=0)
                # this_pos[:, (2, 0)] -= lumin_cent

                this_radii = util.calc_rad(this_pos, i=2, j=0)
                this_gradii = util.calc_rad(this_gpos, i=2, j=0)

                img = util.make_spline_img(this_pos, res, 2, 0, tree,
                                           this_lumin, this_smls,
                                           spline_func=util.cubic_spline,
                                           spline_cut_off=1)

                no_smooth_img = np.histogram2d(this_pos[:, 2], this_pos[:, 0],
                                               range=imgrange, bins=res)

            surf_den = np.sum(img) / (img[img > 0].size * single_pixel_area)

            surf_dens[tag][f].append(surf_den)

            hdr_dict[tag][f].append(util.calc_light_mass_rad(this_gradii,
                                                             this_metals))

            for r in radii_fracs:
                hlr_app_dict[tag][f][r].append(util.get_img_hlr(img,
                                                                apertures,
                                                                app_radii, res,
                                                                csoft, r))

                hlr_pix_dict[tag][f][r].append(
                    util.get_pixel_hlr(img, single_pixel_area, r))

                nosmooth_hlr_app_dict[tag][f][r].append(
                    util.get_img_hlr(no_smooth_img,
                                     apertures,
                                     app_radii, res,
                                     csoft, r))

                nosmooth_hlr_pix_dict[tag][f][r].append(
                    util.get_pixel_hlr(no_smooth_img, single_pixel_area, r))

                hlr_dict[tag][f][r].append(util.calc_light_mass_rad(this_radii,
                                                                    this_lumin,
                                                                    r))

            # Generate SED
            if this_mass > 10 ** 10.5:
                sed = models.generate_SED(model, masses[b: e], this_age,
                                          this_Sz,
                                          tauVs_ISM=reg_dict[f + "tauVs_ISM"][
                                                    b: e],
                                          tauVs_BC=reg_dict[f + "tauVs_BC"][
                                                   b: e],
                                          fesc=0.0,
                                          log10t_BC=reg_dict[f + "log10t_BC"][
                                              ind])

                sed_int[tag][f].append(sed.intrinsic.lnu)
                sed_tot[tag][f].append(sed.total.lnu)
                sed_lam[tag][f].append(sed.lam)

            lumin_dict[tag][f].append(tot_l)
            img_lumin_dict[tag][f].append(np.sum(img))
            nosmooth_img_lumin_dict[tag][f].append(np.sum(img))
            mass_dict[tag][f].append(this_mass)
            gmass_dict[tag][f].append(this_gmass)
            met_dict[tag][f].append(np.sum(this_met))
            gmet_dict[tag][f].append(np.sum(this_metals))
            nstar_dict[tag][f].append(this_nstar)
            img_dict[tag][f].append(img)
            nosmooth_img_dict[tag][f].append(img)

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

        print("There are", len(img_dict[tag][f]), "images")

    hdf = h5py.File(
        "data/flares_sizes_all_{}_{}_{}_{}.hdf5".format(reg, tag, Type,
                                                        orientation),
        "w")

    for f in filters:

        hdr = np.array(hdr_dict[tag][f])
        lumins = np.array(lumin_dict[tag][f])
        img_lumins = np.array(img_lumin_dict[tag][f])
        nosmooth_img_lumins = np.array(nosmooth_img_lumin_dict[tag][f])
        mass = np.array(mass_dict[tag][f])
        gmass = np.array(gmass_dict[tag][f])
        met = np.array(met_dict[tag][f])
        gmet = np.array(gmet_dict[tag][f])
        nstar = np.array(nstar_dict[tag][f])
        imgs = np.array(img_dict[tag][f])
        nosmooth_imgs = np.array(nosmooth_img_dict[tag][f])
        sed_ints = np.array(sed_int[tag][f])
        sed_tots = np.array(sed_tot[tag][f])
        sed_lams = np.array(sed_lam[tag][f])
        surfdens = np.array(surf_dens[tag][f])

        print(imgs.shape)
        print(sed_ints.shape)

        f_group = hdf.create_group(f)

        try:
            dset = f_group.create_dataset("HDR", data=hdr,
                                          dtype=hdr.dtype,
                                          shape=hdr.shape,
                                          compression="gzip")
            dset.attrs["units"] = "pkpc"
        except RuntimeError:
            print("HDR already exists: Overwriting...")
            del f_group["HDR"]
            dset = f_group.create_dataset("HDR", data=hdr,
                                          dtype=hdr.dtype,
                                          shape=hdr.shape,
                                          compression="gzip")
            dset.attrs["units"] = "pkpc"

        try:
            dset = f_group.create_dataset("Surface_Density", data=surfdens,
                                          dtype=surfdens.dtype,
                                          shape=surfdens.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1} / pkpc^2$"
        except RuntimeError:
            print("Surface_Density already exists: Overwriting...")
            del f_group["Surface_Density"]
            dset = f_group.create_dataset("Surface_Density", data=surfdens,
                                          dtype=surfdens.dtype,
                                          shape=surfdens.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1} / pkpc^2$"

        try:
            dset = f_group.create_dataset("Luminosity", data=lumins,
                                          dtype=lumins.dtype,
                                          shape=lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("Luminosity already exists: Overwriting...")
            del f_group["Luminosity"]
            dset = f_group.create_dataset("Luminosity", data=lumins,
                                          dtype=lumins.dtype,
                                          shape=lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("SED_intrinsic", data=sed_ints,
                                          dtype=sed_ints.dtype,
                                          shape=sed_ints.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("SED_intrinsic already exists: Overwriting...")
            del f_group["SED_intrinsic"]
            dset = f_group.create_dataset("SED_intrinsic", data=sed_ints,
                                          dtype=sed_ints.dtype,
                                          shape=sed_ints.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("SED_total", data=sed_tots,
                                          dtype=sed_tots.dtype,
                                          shape=sed_tots.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("SED_total already exists: Overwriting...")
            del f_group["SED_total"]
            dset = f_group.create_dataset("SED_total", data=sed_tots,
                                          dtype=sed_tots.dtype,
                                          shape=sed_tots.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("SED_lambda", data=sed_lams,
                                          dtype=sed_lams.dtype,
                                          shape=sed_lams.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("SED_lambda already exists: Overwriting...")
            del f_group["SED_lambda"]
            dset = f_group.create_dataset("SED_lambda", data=sed_lams,
                                          dtype=sed_lams.dtype,
                                          shape=sed_lams.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("Image_Luminosity", data=img_lumins,
                                          dtype=img_lumins.dtype,
                                          shape=img_lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("Image_Luminosity already exists: Overwriting...")
            del f_group["Image_Luminosity"]
            dset = f_group.create_dataset("Image_Luminosity", data=img_lumins,
                                          dtype=img_lumins.dtype,
                                          shape=img_lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("Image_Luminosity_No_Smooth", data=nosmooth_img_lumins,
                                          dtype=nosmooth_img_lumins.dtype,
                                          shape=nosmooth_img_lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("Image_Luminosity_No_Smooth already exists: Overwriting...")
            del f_group["Image_Luminosity_No_Smooth"]
            dset = f_group.create_dataset("Image_Luminosity_No_Smooth", data=nosmooth_img_lumins,
                                          dtype=nosmooth_img_lumins.dtype,
                                          shape=nosmooth_img_lumins.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("Images already exists: Overwriting...")
            del f_group["Images"]
            dset = f_group.create_dataset("Images", data=imgs,
                                          dtype=imgs.dtype,
                                          shape=imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("No_Smooth_Images", data=nosmooth_imgs,
                                          dtype=nosmooth_imgs.dtype,
                                          shape=nosmooth_imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"
        except RuntimeError:
            print("No_Smooth_Images already exists: Overwriting...")
            del f_group["No_Smooth_Images"]
            dset = f_group.create_dataset("No_Smooth_Images", data=nosmooth_imgs,
                                          dtype=nosmooth_imgs.dtype,
                                          shape=nosmooth_imgs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$erg s^{-1} Hz^{-1}$"

        try:
            dset = f_group.create_dataset("Mass", data=mass,
                                          dtype=mass.dtype,
                                          shape=mass.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"
        except RuntimeError:
            print("Mass already exists: Overwriting...")
            del f_group["Mass"]
            dset = f_group.create_dataset("Mass", data=mass,
                                          dtype=mass.dtype,
                                          shape=mass.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"

        try:
            dset = f_group.create_dataset("Gas_Mass", data=gmass,
                                          dtype=gmass.dtype,
                                          shape=gmass.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"
        except RuntimeError:
            print("Gas_Mass already exists: Overwriting...")
            del f_group["Gas_Mass"]
            dset = f_group.create_dataset("Gas_Mass", data=gmass,
                                          dtype=gmass.dtype,
                                          shape=gmass.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"

        try:
            dset = f_group.create_dataset("Gas_Metal_Mass", data=gmet,
                                          dtype=gmet.dtype,
                                          shape=gmet.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"
        except RuntimeError:
            print("Gas_Metal_Mass already exists: Overwriting...")
            del f_group["Gas_Metal_Mass"]
            dset = f_group.create_dataset("Gas_Metal_Mass", data=gmet,
                                          dtype=gmet.dtype,
                                          shape=gmet.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"

        try:
            dset = f_group.create_dataset("Star_Metal_Mass", data=met,
                                          dtype=met.dtype,
                                          shape=met.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"
        except RuntimeError:
            print("Star_Metal_Mass already exists: Overwriting...")
            del f_group["Star_Metal_Mass"]
            dset = f_group.create_dataset("Star_Metal_Mass", data=met,
                                          dtype=met.dtype,
                                          shape=met.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$M_\odot$"

        try:
            dset = f_group.create_dataset("nStar", data=nstar,
                                          dtype=nstar.dtype,
                                          shape=nstar.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"
        except RuntimeError:
            print("nStar already exists: Overwriting...")
            del f_group["nStar"]
            dset = f_group.create_dataset("nStar", data=nstar,
                                          dtype=nstar.dtype,
                                          shape=nstar.shape,
                                          compression="gzip")
            dset.attrs["units"] = "None"

        for r in radii_fracs:

            hlrs = np.array(hlr_dict[tag][f][r])
            hlrs_app = np.array(hlr_app_dict[tag][f][r])
            hlrs_pix = np.array(hlr_pix_dict[tag][f][r])

            nosmooth_hlrs = np.array(nosmooth_hlr_dict[tag][f][r])
            nosmooth_hlrs_app = np.array(nosmooth_hlr_app_dict[tag][f][r])
            nosmooth_hlrs_pix = np.array(nosmooth_hlr_pix_dict[tag][f][r])

            try:
                dset = f_group.create_dataset("HLR_%.1f" % r, data=hlrs,
                                              dtype=hlrs.dtype,
                                              shape=hlrs.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"
            except RuntimeError:
                print("HLR_%.1f" % r, "already exists: Overwriting...")
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
                print("HLR_Aperture_%.1f" % r,
                      "already exists: Overwriting...")
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
                print("HLR_Pixel_%.1f" % r, "already exists: Overwriting...")
                del f_group["HLR_Pixel_%.1f" % r]
                dset = f_group.create_dataset("HLR_Pixel_%.1f" % r,
                                              data=hlrs_pix,
                                              dtype=hlrs_pix.dtype,
                                              shape=hlrs_pix.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"

            try:
                dset = f_group.create_dataset("HLR_Aperture_%.1f_No_Smooth" % r,
                                              data=nosmooth_hlrs_app,
                                              dtype=nosmooth_hlrs_app.dtype,
                                              shape=nosmooth_hlrs_app.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"
            except RuntimeError:
                print("HLR_Aperture_%.1f_No_Smooth" % r,
                      "already exists: Overwriting...")
                del f_group["HLR_Aperture_%.1f_No_Smooth" % r]
                dset = f_group.create_dataset("HLR_Aperture_%.1f_No_Smooth" % r,
                                              data=nosmooth_hlrs_app,
                                              dtype=nosmooth_hlrs_app.dtype,
                                              shape=nosmooth_hlrs_app.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"

            try:
                dset = f_group.create_dataset("HLR_Pixel_%.1f_No_Smooth" % r,
                                              data=nosmooth_hlrs_pix,
                                              dtype=nosmooth_hlrs_pix.dtype,
                                              shape=nosmooth_hlrs_pix.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"
            except RuntimeError:
                print("HLR_Pixel_%.1f_No_Smooth" % r, "already exists: Overwriting...")
                del f_group["HLR_Pixel_%.1f_No_Smooth" % r]
                dset = f_group.create_dataset("HLR_Pixel_%.1f_No_Smooth" % r,
                                              data=nosmooth_hlrs_pix,
                                              dtype=nosmooth_hlrs_pix.dtype,
                                              shape=nosmooth_hlrs_pix.shape,
                                              compression="gzip")
                dset.attrs["units"] = "$\mathrm{pkpc}$"

    hdf.close()
