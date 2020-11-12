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


# snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
#          '006_z009p000', '007_z008p000', '008_z007p000',
#          '009_z006p000', '010_z005p000', '011_z004p770']
snaps = ['007_z008p000', ]

# Define filter
filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV')

# Define dictionaries for results
hlr_dict = {}
hlr_app_dict = {}
lumin_dict = {}
mass_dict = {}

# Set orientation
orientation = sys.argv[1]

# Define luminosity and dust model types
Type = sys.argv[2]
extinction = 'default'

# Set mass limit
masslim = 10 ** 7

for tag in snaps:

    z_str = tag.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    hlr_dict.setdefault(tag, {})
    hlr_app_dict.setdefault(tag, {})
    lumin_dict.setdefault(tag, {})
    mass_dict.setdefault(tag, {})

    # Kappa with DTM 0.0795, BC_fac=1., without 0.0063 BC_fac=1.25

    lumin_dicts = phot.get_lum_all(kappa=0.0795, tag=tag, BC_fac=1,
                                   IMF='Chabrier_300',
                                   bins=np.arange(-24, -16, 0.5), inp='FLARES',
                                   LF=False, filters=filters, Type=Type,
                                   log10t_BC=7., extinction=extinction,
                                   orientation=orientation, numThreads=8,
                                   masslim=masslim)

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

    # Define range and extent for the images
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
    imgextent = [-width / 2, width / 2, -width / 2, width / 2]

    # Set up aperture objects
    positions = [(res / 2, res / 2)]
    app_radii = np.linspace(0.001, res / 4, 100)
    apertures = [CircularAperture(positions, r=r) for r in app_radii]
    app_radii *= csoft

    for num, reg_dict in enumerate(lumin_dicts):

        print("Processing galaxies in results", num)

        poss = reg_dict["coords"] * 10 ** 3
        smls = reg_dict["smls"] * 10 ** 3
        masses = reg_dict["masses"]
        begin = reg_dict["begin"]
        end = reg_dict["end"]

        for f in filters:

            hlr_dict[tag].setdefault(f, [])
            hlr_app_dict[tag].setdefault(f, [])
            lumin_dict[tag].setdefault(f, [])
            mass_dict[tag].setdefault(f, [])

            for ind in range(len(begin)):

                b, e = begin[ind], end[ind]

                this_pos = poss[:, b: e].T
                this_lumin = reg_dict[f][b: e]
                this_smls = smls[b: e]
                this_mass = np.nansum(masses[b: e])
                # this_smls = np.full_like(this_lumin, csoft / 2.355)

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

                # hlr_app_dict[tag][f].append(util.get_img_hlr(img,
                #                                              apertures,
                #                                              app_radii, res,
                #                                              csoft))

                hlr_dict[tag][f].append(util.calc_light_mass_rad(this_radii,
                                                                 this_lumin))

                lumin_dict[tag][f].append(tot_l)
                mass_dict[tag][f].append(this_mass)

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
    hdf = h5py.File("flares_sizes.hdf5", "r+")
except OSError:
    hdf = h5py.File("flares_sizes.hdf5", "w")

try:
    type_group = hdf[Type]
except KeyError:
    type_group = hdf.create_group(Type)

try:
    orientation_group = type_group[orientation]
except KeyError:
    orientation_group = type_group.create_group(orientation)

for snap in snaps:
    for f in filters:

        hlrs = np.array(hlr_dict[snap][f])
        hlrs_app = np.array(hlr_app_dict[snap][f])
        lumins = np.array(lumin_dict[snap][f])
        mass = np.array(mass_dict[snap][f])

        try:
            snap_group = orientation_group[snap]
        except KeyError:
            snap_group = orientation_group.create_group(snap)

        try:
            f_group = snap_group[f]
        except KeyError:
            f_group = snap_group.create_group(f)

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

        try:
            dset = f_group.create_dataset("HLR", data=hlrs,
                                          dtype=hlrs.dtype,
                                          shape=hlrs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            del f_group["HLR"]
            dset = f_group.create_dataset("HLR", data=hlrs,
                                          dtype=hlrs.dtype,
                                          shape=hlrs.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

        try:
            dset = f_group.create_dataset("HLR_Aperture", data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"
        except RuntimeError:
            del f_group["HLR_Aperture"]
            dset = f_group.create_dataset("HLR_Aperture", data=hlrs_app,
                                          dtype=hlrs_app.dtype,
                                          shape=hlrs_app.shape,
                                          compression="gzip")
            dset.attrs["units"] = "$\mathrm{pkpc}$"

for f in filters:

    fit_lumins = np.logspace(28, 31, 1000)

    if len(snaps) == 9:

        axlims_x = []
        axlims_y = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])
            print(hlrs, lumins)
            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            print(hlrs, lumins)
            try:
                cbar = ax.hexbin(lumins, hlrs / (csoft / (1 + z)),
                                 gridsize=100,
                                 mincnt=1, xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                if lumins.size > 10:
                    plot_meidan_stat(lumins, hlrs / (csoft / (1 + z)), ax,
                                     lab='REF',
                                     color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/\epsilon$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(10 ** -1.1, 10 ** 2.2)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        handles, labels = ax6.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="")

        fig.savefig('plots/HalfLightRadius_' + f + '_soft_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.2f.png' % np.log10(masslim),
                    bbox_inches='tight')

        plt.close(fig)

        axlims_x = []
        axlims_y = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        handles, labels = ax6.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="bottom right")

        fig.savefig('plots/HalfLightRadius_' + f + '_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.2f.png' % np.log10(masslim), bbox_inches='tight')

        plt.close(fig)

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            fig.savefig('plots/HalfLightRadius_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.2f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

        axlims_x = []
        axlims_y = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            try:
                cbar = ax.hexbin(lumins, hlrs / (csoft / (1 + z)),
                                 gridsize=100,
                                 mincnt=1, xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                if lumins.size > 10:
                    plot_meidan_stat(lumins, hlrs / (csoft / (1 + z)), ax,
                                     lab='REF',
                                     color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/\epsilon$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(10 ** -1.1, 10 ** 2.2)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        handles, labels = ax6.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="bottom right")

        fig.savefig('plots/HalfLightRadiusAperture_' + f + '_soft_'
                    + orientation + '_' + Type + "_" + extinction
                    + "_" + '%.2f.png' % np.log10(masslim),
                    bbox_inches='tight')

        plt.close(fig)

        axlims_x = []
        axlims_y = []

        # Set up plot
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 6)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        for ax, snap, (i, j) in zip(
                [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
                snaps,
                [(0, 0), (0, 1), (0, 2),
                 (1, 0), (1, 1), (1, 2),
                 (2, 0), (2, 1), (2, 2)]):

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds] * 1000
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                ax.fill_between(fit_lumins, low, up,
                                color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axlims_x.extend(ax.get_xlim())
            axlims_y.extend(ax.get_ylim())

            # Label axes
            if i == 2:
                ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            if j == 0:
                ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            ax.set_xlim(np.min(axlims_x), np.max(axlims_x))
            ax.set_ylim(np.min(axlims_y), np.max(axlims_y))
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Remove axis labels
        ax1.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax2.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax3.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax4.tick_params(axis='x', top=False, bottom=False,
                        labeltop=False, labelbottom=False)
        ax5.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax6.tick_params(axis='both', left=False, top=False,
                        right=False, bottom=False,
                        labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        ax8.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
        ax9.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)

        handles, labels = ax6.get_legend_handles_labels()
        ax1.legend(handles, labels, loc="bottom right")

        fig.savefig('plots/HalfLightRadiusAperture_'
                    + f + '_' + orientation + '_'
                    + Type + "_" + extinction + "_"
                    + '%.2f.png' % np.log10(masslim), bbox_inches='tight')

        plt.close(fig)

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_app_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(loc="")

            fig.savefig('plots/HalfLightRadiusAperture_'
                        + f + '_' + str(z) + '_' + orientation
                        + '_' + Type + "_" + extinction + "_"
                        + '%.2f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

    else:

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            hlrs = np.array(hlr_dict[snap][f])
            lumins = np.array(lumin_dict[snap][f])
            mass = np.array(mass_dict[snap][f])

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-18),
                                                   lumins < 10 ** 50))
            lumins = lumins[okinds]
            hlrs = hlrs[okinds]
            mass = mass[okinds]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
                                 xscale='log', yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError:
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(fit_lumins, fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                # ax.fill_between(fit_lumins, low, up,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(loc="bottom right")

            fig.savefig('plots/HalfLightRadius_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.2f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            try:
                cbar = ax.hexbin(lum_to_M(lumins), hlrs, gridsize=50, mincnt=1,
                                 yscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError as e:
                print(e)
                continue

            if int(z) in [6, 7, 8, 9]:
                fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
                               kawa_params['beta'][int(z)])
                up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                  kawa_params['beta'][int(z)],
                                  kawa_up_params['r_0'][int(z)],
                                  kawa_up_params['beta'][int(z)], uplow="low")
                low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
                                   kawa_params['beta'][int(z)],
                                   kawa_low_params['r_0'][int(z)],
                                   kawa_low_params['beta'][int(z)],
                                   uplow="low")
                ax.plot(lum_to_M(fit_lumins), fit,
                        linestyle='dashed', color='k', alpha=0.9, zorder=2,
                        label="Kawamata+18")
                # ax.fill_between(lum_to_M(fit_lumins), up, low,
                #                 color='k', alpha=0.4, zorder=1)

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_{UV}$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            ax.legend(loc="bottom left")

            fig.savefig('plots/HalfLightRadius_AbMag_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.2f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            try:
                cbar = ax.hexbin(mass, hlrs, gridsize=50, mincnt=1,
                                 yscale='log', xscale='log',
                                 norm=LogNorm(), linewidths=0.2,
                                 cmap='viridis')
                # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            except ValueError as e:
                print(e)
                continue

            ax.text(0.8, 0.1, f'$z={z}$',
                    bbox=dict(boxstyle="round,pad=0.3", fc='w',
                              ec="k", lw=1, alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            # Label axes
            ax.set_xlabel(r'$M_\star/M_\odot$')
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')

            fig.savefig('plots/HalfLightRadius_Mass_' + f + '_' + str(z) + '_'
                        + orientation + '_' + Type + "_" + extinction + "_"
                        + '%.2f.png' % np.log10(masslim),
                        bbox_inches='tight')

            plt.close(fig)

            # hlrs = np.array(hlr_app_dict[snap][f])
            # lumins = np.array(lumin_dict[snap][f])
            #
            # okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
            #                         np.logical_and(lumins > M_to_lum(-18),
            #                                        lumins < 10 ** 50))
            # lumins = lumins[okinds]
            # hlrs = hlrs[okinds]
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # try:
            #     cbar = ax.hexbin(lumins, hlrs, gridsize=50, mincnt=1,
            #                      xscale='log', yscale='log',
            #                      norm=LogNorm(), linewidths=0.2, cmap='viridis')
            #     # plot_meidan_stat(lumins, hlrs * 10**3, ax, lab='REF', color='r')
            # except ValueError:
            #     continue
            #
            # if int(z) in [6, 7, 8, 9]:
            #     fit = kawa_fit(fit_lumins, kawa_params['r_0'][int(z)],
            #                    kawa_params['beta'][int(z)])
            #     up = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
            #                       kawa_params['beta'][int(z)],
            #                       kawa_up_params['r_0'][int(z)],
            #                       kawa_up_params['beta'][int(z)], uplow="low")
            #     low = kawa_fit_err(fit, fit_lumins, kawa_params['r_0'][int(z)],
            #                        kawa_params['beta'][int(z)],
            #                        kawa_low_params['r_0'][int(z)],
            #                        kawa_low_params['beta'][int(z)],
            #                        uplow="low")
            #     ax.plot(fit_lumins, fit,
            #             linestyle='dashed', color='k', alpha=0.9, zorder=2,
            #             label="Kawamata+18")
            #     ax.fill_between(fit_lumins, low, up,
            #                     color='k', alpha=0.4, zorder=1)
            #
            # ax.text(0.8, 0.1, f'$z={z}$',
            #         bbox=dict(boxstyle="round,pad=0.3", fc='w',
            #                   ec="k", lw=1, alpha=0.8),
            #         transform=ax.transAxes, horizontalalignment='right',
            #         fontsize=8)
            #
            # # Label axes
            # ax.set_xlabel(r'$L_{FUV}/$ [erg $/$ s $/$ Hz]')
            # ax.set_ylabel('$R_{1/2}/ [pkpc]$')
            #
            # fig.savefig('plots/HalfLightRadiusAperture_'
            #             + f + '_' + str(z) + '_' + orientation
            #             + '_' + Type + "_" + extinction + "_"
            #             + '%.2f.png' % np.log10(masslim),
            #             bbox_inches='tight')
            #
            # plt.close(fig)
