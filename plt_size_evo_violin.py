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
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import h5py
import sys
import pandas as pd
import weighted
from matplotlib.cbook import violin_stats
import statsmodels.api as sm
from scipy.optimize import curve_fit

sns.set_context("paper")
sns.set_style('whitegrid')


def vdensity_with_weights(weights):
    ''' Outer function allows innder function access to weights. Matplotlib
    needs function to take in data and coords, so this seems like only way
    to 'pass' custom density function a set of weights '''

    def vdensity(data, coords):
        ''' Custom matplotlib weighted violin stats function '''
        # Using weights from closure, get KDE fomr statsmodels
        weighted_cost = sm.nonparametric.KDEUnivariate(data)
        weighted_cost.fit(fft=False, weights=weights)

        # Return y-values for graph of KDE by evaluating on coords
        return weighted_cost.evaluate(coords)

    return vdensity


def custom_violin_stats(data, weights):
    # Get weighted median and mean (using weighted module for median)
    median = weighted.quantile_1D(data, weights, 0.5)
    pcent_16 = weighted.quantile_1D(data, weights, 0.16)
    pcent_84 = weighted.quantile_1D(data, weights, 0.84)
    mean, sumw = np.ma.average(data, weights=list(weights), returned=True)

    # Use matplotlib violin_stats, which expects a function that takes in data and coords
    # which we get from closure above
    results = violin_stats(data, vdensity_with_weights(weights))

    # Update result dictionary with our updated info
    results[0][u"mean"] = mean
    results[0][u"median"] = median
    results[0][u"pcent_16"] = pcent_16
    results[0][u"pcent_84"] = pcent_84

    # No need to do this, since it should be populated from violin_stats
    # results[0][u"min"] =  np.min(data)
    # results[0][u"max"] =  np.max(data)

    return results


def fit(z, C, m):
    return C * (1 + z) ** -m


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


def r_from_surf_den(lum, s_den):
    return np.sqrt(lum / (s_den * np.pi))


def lum_from_surf_den_R(r, s_den):
    return s_den * np.pi * r ** 2


df = pd.read_csv("HighzSizes/All.csv")

papers = df["Paper"].values
mags = df["Magnitude"].values
r_es_arcs = df["R_e"].values
r_es_type = df["R_e (Unit)"].values
mag_type = df["Magnitude Type"].values
zs = df["Redshift"].values

# Define pixel resolutions
wfc3 = 0.13
nircam_short = 0.031
nircam_long = 0.063

# Convert to physical kiloparsecs
r_es = np.zeros(len(papers))
for (ind, r), z in zip(enumerate(r_es_arcs), zs):
    if r_es_type[ind] == "kpc":
        r_es[ind] = r
    else:
        r_es[ind] = r / cosmo.arcsec_per_kpc_proper(z).value
    if mags[ind] < 0:
        mags[ind] = M_to_m(mags[ind], cosmo, z)

cmap = mpl.cm.get_cmap("winter")
norm = plt.Normalize(vmin=0, vmax=1)

labels = {"G11": "Grazian+2011",
          "G12": "Grazian+2012",
          "C16": "Calvi+2016",
          "K18": "Kawamata+2018",
          "MO18": "Morishita+2018",
          "B19": "Bridge+2019",
          "O16": "Oesch+2016",
          "S18": "Salmon+2018",
          "H20": "Holwerda+2020"}
markers = {"G11": "s", "G12": "v", "C16": "D",
           "K18": "o", "M18": "X", "MO18": "o",
           "B19": "^", "O16": "P", "S18": "<", "H20": "*"}
colors = {}
for key, col in zip(markers.keys(), np.linspace(0, 1, len(markers.keys()))):
    colors[key] = cmap(norm(col))


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ https://stackoverflow.com/questions/21844024/
        weighted-percentile-using-numpy

    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


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
extinction = 'default'

snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000',
         '009_z006p000', '010_z005p000']

# Define filter
filters = ['FAKE.TH.' + f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]

csoft = 0.001802390 / (0.6777) * 1e3

nlim = 10 ** 8

hlr_dict = {}
hdr_dict = {}
hlr_app_dict = {}
hlr_pix_dict = {}
lumin_dict = {}
img_lumin_dict = {}
weight_dict = {}
mass_dict = {}

intr_hlr_dict = {}
intr_hlr_app_dict = {}
intr_hlr_pix_dict = {}
intr_lumin_dict = {}
intr_img_lumin_dict = {}
intr_weight_dict = {}

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

    hdf = h5py.File(
        "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Total",
                                                    orientation),
        "r")

    hlr_dict.setdefault(snap, {})
    hdr_dict.setdefault(snap, {})
    hlr_app_dict.setdefault(snap, {})
    hlr_pix_dict.setdefault(snap, {})
    lumin_dict.setdefault(snap, {})
    img_lumin_dict.setdefault(snap, {})
    weight_dict.setdefault(snap, {})
    mass_dict.setdefault(snap, {})

    for f in filters:
        hlr_dict[snap].setdefault(f, [])
        hdr_dict[snap].setdefault(f, [])
        hlr_app_dict[snap].setdefault(f, [])
        hlr_pix_dict[snap].setdefault(f, [])
        lumin_dict[snap].setdefault(f, [])
        img_lumin_dict[snap].setdefault(f, [])
        weight_dict[snap].setdefault(f, [])
        mass_dict[snap].setdefault(f, [])

        masses = hdf[f]["Mass"][...]
        okinds = masses > nlim

        print(reg, snap, f, masses[okinds].size)

        hlr_dict[snap][f].extend(hdf[f]["HLR_0.5"][...][okinds])
        hdr_dict[snap][f].extend(hdf[f]["HDR"][...][okinds])
        hlr_app_dict[snap][f].extend(hdf[f]["HLR_Aperture_0.5"][...][okinds])
        hlr_pix_dict[snap][f].extend(hdf[f]["HLR_Pixel_0.5"][...][okinds])
        lumin_dict[snap][f].extend(hdf[f]["Luminosity"][...][okinds])
        img_lumin_dict[snap][f].extend(hdf[f]["Image_Luminosity"][...][okinds])
        weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                            weights[int(reg)]))
        mass_dict[snap][f].extend(masses[okinds])

    hdf.close()

    hdf = h5py.File("data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                "Intrinsic",
                                                                orientation),
                    "r")

    intr_hlr_dict.setdefault(snap, {})
    intr_hlr_app_dict.setdefault(snap, {})
    intr_hlr_pix_dict.setdefault(snap, {})
    intr_img_lumin_dict.setdefault(snap, {})
    intr_lumin_dict.setdefault(snap, {})
    intr_weight_dict.setdefault(snap, {})

    for f in filters:
        intr_hlr_dict[snap].setdefault(f, [])
        intr_hlr_app_dict[snap].setdefault(f, [])
        intr_hlr_pix_dict[snap].setdefault(f, [])
        intr_lumin_dict[snap].setdefault(f, [])
        intr_img_lumin_dict[snap].setdefault(f, [])
        intr_weight_dict[snap].setdefault(f, [])

        masses = hdf[f]["Mass"][...]
        okinds = masses > nlim

        print(reg, snap, f, masses[okinds].size)

        intr_hlr_dict[snap][f].extend(hdf[f]["HLR_0.5"][...][okinds])
        intr_hlr_app_dict[snap][f].extend(
            hdf[f]["HLR_Aperture_0.5"][...][okinds])
        intr_hlr_pix_dict[snap][f].extend(
            hdf[f]["HLR_Pixel_0.5"][...][okinds])
        intr_lumin_dict[snap][f].extend(
            hdf[f]["Luminosity"][...][okinds])
        intr_img_lumin_dict[snap][f].extend(
            hdf[f]["Image_Luminosity"][...][okinds])
        intr_weight_dict[snap][f].extend(np.full(masses[okinds].size,
                                                 weights[int(reg)]))

    hdf.close()
for mtype in ["pix", "part", "app"]:
    for f in filters:

        print("Plotting for:")
        print("Orientation =", orientation)
        print("Filter =", f)

        hlr = []
        intr_hlr = []
        hdr = []
        ws = []
        plt_z = []
        ms = []

        for snap in snaps:

            z_str = snap.split('z')[1].split('p')
            z = float(z_str[0] + '.' + z_str[1])

            if mtype == "part":
                hlrs = np.array(hlr_dict[snap][f])
                intr_hlrs = np.array(intr_hlr_dict[snap][f])
                lumins = np.array(lumin_dict[snap][f])
                intr_lumins = np.array(intr_lumin_dict[snap][f])
            elif mtype == "app":
                hlrs = np.array(hlr_app_dict[snap][f])
                intr_hlrs = np.array(intr_hlr_app_dict[snap][f])
                lumins = np.array(img_lumin_dict[snap][f])
                intr_lumins = np.array(intr_img_lumin_dict[snap][f])
            else:
                hlrs = np.array(hlr_pix_dict[snap][f])
                intr_hlrs = np.array(intr_hlr_pix_dict[snap][f])
                lumins = np.array(img_lumin_dict[snap][f])
                intr_lumins = np.array(intr_img_lumin_dict[snap][f])
            m = np.array(mass_dict[snap][f])
            hdrs = np.array(hdr_dict[snap][f])
            w = np.array(weight_dict[snap][f])

            if len(w) == 0:
                continue

            okinds = np.logical_and(hlrs / (csoft / (1 + z)) > 10 ** -1,
                                    np.logical_and(lumins > M_to_lum(-12),
                                                   lumins < 10 ** 50))

            hlr.append(hlrs[okinds])
            hdr.append(hdrs[okinds])
            intr_hlr.append(intr_hlrs[okinds])
            ws.append(w[okinds])
            ms.append(m[okinds])

            # quants = weighted_quantile(hlrs, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # hlr_med.append(quants[1])
            # hlr_16.append(quants[0])
            # hlr_84.append(quants[2])
            #
            # quants = weighted_quantile(hlrs_app, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # hlr_med_app.append(quants[1])
            # hlr_16_app.append(quants[0])
            # hlr_84_app.append(quants[2])
            #
            # quants = weighted_quantile(hlrs_pix, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # hlr_med_pix.append(quants[1])
            # hlr_16_pix.append(quants[0])
            # hlr_84_pix.append(quants[2])
            #
            # quants = weighted_quantile(intr_hlrs, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # intr_hlr_med.append(quants[1])
            # intr_hlr_16.append(quants[0])
            # intr_hlr_84.append(quants[2])
            #
            # quants = weighted_quantile(intr_hlrs_app, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # intr_hlr_med_app.append(quants[1])
            # intr_hlr_16_app.append(quants[0])
            # intr_hlr_84_app.append(quants[2])
            #
            # quants = weighted_quantile(intr_hlrs_pix, [0.16, 0.5, 0.84], sample_weight=w,
            #                            values_sorted=False, old_style=False)
            #
            # intr_hlr_med_pix.append(quants[1])
            # intr_hlr_16_pix.append(quants[0])
            # intr_hlr_84_pix.append(quants[2])

            plt_z.append(z)

        fitting_hlrs = []
        fitting_zs = []
        fitting_ws = []
        fitting_ms = []

        for i in range(len(hlr)):
            fitting_zs.extend(np.full(len(hlr[i]), plt_z[i]))
            fitting_hlrs.extend(hlr[i])
            fitting_ws.extend(ws[i])
            fitting_ms.extend(ms[i])

        fitting_hlrs = np.array(fitting_hlrs)
        fitting_zs = np.array(fitting_zs)
        fitting_ws = np.array(fitting_ws)
        fitting_ms = np.array(fitting_ms)

        soft = []
        for z in plt_z:

            if z <= 2.8:
                soft.append(0.000474390 / 0.6777 * 1e3)
            else:
                soft.append(0.001802390 / (0.6777 * (1 + z)) * 1e3)

        legend_elements = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy()
        # ax.plot(plt_z, soft, color="k", linestyle="--", label="Softening")
        for i in range(len(ws)):

            vpstats1 = custom_violin_stats(hlr[i], ws[i])
            vplot = ax.violin(vpstats1, positions=[plt_z[i]],
                              vert=True,
                              showmeans=True,
                              showextrema=True,
                              showmedians=True)

            # Make all the violin statistics marks red:
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                vp = vplot[partname]
                vp.set_edgecolor("k")
                vp.set_linewidth(1)

            # Make the violin body blue with a red border:
            for vp in vplot['bodies']:
                vp.set_facecolor("r")
                vp.set_edgecolor("r")
                vp.set_linewidth(1)
                vp.set_alpha(0.5)

        # try:
        # ax.fill_between(plt_z, intr_hlr_16, intr_hlr_84, color="r", alpha=0.4)
        # ax.plot(plt_z, intr_hlr_med, color="r", marker="D", linestyle="--")
        # legend_elements.append(
        #     Line2D([0], [0], color="r", linestyle="--", label="Intrinsic"))
        #
        # ax.fill_between(plt_z, hlr_16, hlr_84, color="g", alpha=0.4)
        # ax.plot(plt_z, hlr_med, color="g", marker="^", linestyle="-")
        # legend_elements.append(
        #     Line2D([0], [0], color="g", linestyle="-",
        #            label="Attenuated"))
        # except ValueError as e:
        #     print(e)
        #     continue

        # for p in labels.keys():
        #
        #     okinds = papers == p
        #     plt_r_es = r_es[okinds]
        #     plt_zs = zs[okinds]
        #
        #     if plt_zs.size == 0:
        #         continue
        #
        #     legend_elements.append(
        #         Line2D([0], [0], marker=markers[p], color='w',
        #                label=labels[p], markerfacecolor=colors[p],
        #                markersize=8, alpha=0.7))
        #
        #     ax.scatter(plt_zs, plt_r_es,
        #                marker=markers[p], label=labels[p], s=17,
        #                color=colors[p], alpha=0.7)

        okinds2 = fitting_ms > 10 ** 9

        popt, pcov = curve_fit(fit, fitting_zs, fitting_hlrs,
                               p0=(1, 0.5), sigma=fitting_ws)

        popt1, pcov1 = curve_fit(fit, fitting_zs[okinds2],
                                 fitting_hlrs[okinds2],
                                 p0=(1, 0.5), sigma=fitting_ws[okinds2])

        fit_plt_zs = np.linspace(12, 4.5, 1000)

        print("--------------", "Total", "All", mtype, f, "--------------")
        print("C=", popt[0], "+/-", np.sqrt(pcov[0, 0]))
        print("m=", popt[1], "+/-", np.sqrt(pcov[1, 1]))
        print(pcov)
        print("--------------", "Total", "Massive", mtype, f, "--------------")
        print("C=", popt1[0], "+/-", np.sqrt(pcov1[0, 0]))
        print("m=", popt1[1], "+/-", np.sqrt(pcov1[1, 1]))
        print(pcov)
        print("----------------------------------------------------------")

        ax.plot(fit_plt_zs, fit(fit_plt_zs, popt[0], popt[1]),
                linestyle="--", color="k")

        legend_elements.append(Line2D([0], [0], color='k', label="All",
                                      linestyle="--"))

        ax.plot(fit_plt_zs, fit(fit_plt_zs, popt1[0], popt1[1]),
                linestyle="dotted", color="k")

        legend_elements.append(Line2D([0], [0], color='k',
                                      label="$M_\star/M\odot$",
                                      linestyle="dotted"))

        # Label axes
        ax.set_xlabel(r'$z$')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')

        ax.tick_params(axis='x', which='minor', bottom=True)

        ax.set_xlim(4.5, 11.5)
        ax.set_ylim(10 ** -1.5, 10 ** 1.5)

        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

        fig.savefig(
            'plots/Violin_HalfLightRadius_evolution_' + mtype + '_' + f + '_'
            + orientation + "_" + extinction + "_"
            + '%d.png' % nlim,
            bbox_inches='tight')

        plt.close(fig)

        legend_elements = []

        fig = plt.figure(figsize=(5, 9))
        gs = gridspec.GridSpec(3, 1)
        gs.update(wspace=0.0, hspace=0.0)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])

        for ax in [ax1, ax2, ax3]:

            if ax != ax3:
                ax.tick_params(axis='x', top=False, bottom=False,
                               labeltop=False, labelbottom=False)
            ax.semilogy()

        med_hlr = []
        med_inthlr = []
        med_hdr = []
        hlr_16 = []
        inthlr_16 = []
        hdr_16 = []
        hlr_84 = []
        inthlr_84 = []
        hdr_84 = []
        for i in range(len(ws)):

            vpstats1 = custom_violin_stats(hlr[i], ws[i])
            med_hlr.append(vpstats1[0]["median"])
            hlr_16.append(vpstats1[0]["pcent_16"])
            hlr_84.append(vpstats1[0]["pcent_84"])
            vplot = ax1.violin(vpstats1, positions=[plt_z[i]],
                               vert=True,
                               showmeans=True,
                               showextrema=True,
                               showmedians=True)

            # Make all the violin statistics marks red:
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                vp = vplot[partname]
                vp.set_edgecolor("k")
                vp.set_linewidth(1)

            # Make the violin body blue with a red border:
            for vp in vplot['bodies']:
                vp.set_facecolor("r")
                vp.set_edgecolor("r")
                vp.set_linewidth(1)
                vp.set_alpha(0.5)

        for i in range(len(ws)):

            vpstats1 = custom_violin_stats(intr_hlr[i], ws[i])
            med_inthlr.append(vpstats1[0]["median"])
            inthlr_16.append(vpstats1[0]["pcent_16"])
            inthlr_84.append(vpstats1[0]["pcent_84"])
            vplot = ax2.violin(vpstats1, positions=[plt_z[i]],
                               vert=True,
                               showmeans=True,
                               showextrema=True,
                               showmedians=True)

            # Make all the violin statistics marks red:
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                vp = vplot[partname]
                vp.set_edgecolor("k")
                vp.set_linewidth(1)

            # Make the violin body blue with a red border:
            for vp in vplot['bodies']:
                vp.set_facecolor("g")
                vp.set_edgecolor("g")
                vp.set_linewidth(1)
                vp.set_alpha(0.5)

        for i in range(len(ws)):

            vpstats1 = custom_violin_stats(hdr[i], ws[i])
            med_hdr.append(vpstats1[0]["median"])
            hdr_16.append(vpstats1[0]["pcent_16"])
            hdr_84.append(vpstats1[0]["pcent_84"])
            vplot = ax3.violin(vpstats1, positions=[plt_z[i]],
                               vert=True,
                               showmeans=True,
                               showextrema=True,
                               showmedians=True)

            # Make all the violin statistics marks red:
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                vp = vplot[partname]
                vp.set_edgecolor("k")
                vp.set_linewidth(1)

            # Make the violin body blue with a red border:
            for vp in vplot['bodies']:
                vp.set_facecolor("m")
                vp.set_edgecolor("m")
                vp.set_linewidth(1)
                vp.set_alpha(0.5)

        for ax in [ax1, ax2, ax3]:
            ax.plot(plt_z, soft, color="k", linestyle="--", label="Softening")
            ax.plot(plt_z, med_inthlr, color="g", marker="s", linestyle="-")
            ax.plot(plt_z, med_hlr, color="r", marker="^", linestyle="-")
            ax.plot(plt_z, med_hdr, color="m", marker="D", linestyle="-")

        legend_elements.append(
            Line2D([0], [0], color="g", linestyle="-", marker="s",
                   label="Median Intrinsic"))
        legend_elements.append(
            Line2D([0], [0], color="r", linestyle="-", marker="^",
                   label="Median Attenuated"))
        legend_elements.append(
            Line2D([0], [0], color="m", linestyle="-", marker="D",
                   label="Median Dust"))
        legend_elements.append(
            Line2D([0], [0], color="k", linestyle="--",
                   label="Softening"))

        # Label axes
        ax3.set_xlabel(r'$z$')
        for ax in [ax1, ax2, ax3]:
            ax.set_ylabel('$R_{1/2}/ [pkpc]$')
            ax.set_xlim(4.5, 11.5)
            ax.set_ylim(10 ** -1.5, 10 ** 1.5)

        ax3.tick_params(axis='x', which='minor', bottom=True)

        ax3.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)

        fig.savefig(
            'plots/ViolinComp_HalfLightRadius_evolution_' + mtype + '_' + f + '_'
            + orientation + "_" + extinction + "_"
            + '%d.png' % nlim,
            bbox_inches='tight')

        plt.close(fig)

        legend_elements = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogy()

        ax.plot(plt_z, soft, color="k", linestyle="--", label="Softening")
        ax.fill_between(plt_z, hdr_16, hdr_84, facecolor="none", hatch="X",
                        edgecolor="m")
        ax.fill_between(plt_z, inthlr_16, inthlr_84, color="g", alpha=0.4)
        ax.fill_between(plt_z, hlr_16, hlr_84, color="r", alpha=0.4)
        ax.plot(plt_z, med_hdr, color="m", marker="D", linestyle="-")
        ax.plot(plt_z, med_inthlr, color="g", marker="s", linestyle="-")
        ax.plot(plt_z, med_hlr, color="r", marker="^", linestyle="-")

        legend_elements.append(
            Line2D([0], [0], color="g", linestyle="-", marker="s",
                   label="Median Intrinsic"))
        legend_elements.append(
            Line2D([0], [0], color="r", linestyle="-", marker="^",
                   label="Median Attenuated"))
        legend_elements.append(
            Line2D([0], [0], color="m", linestyle="-", marker="D",
                   label="Median Dust"))
        legend_elements.append(
            Line2D([0], [0], color="k", linestyle="--",
                   label="Softening"))

        for p in labels.keys():

            okinds = papers == p
            plt_r_es = r_es[okinds]
            plt_zs = zs[okinds]

            if plt_zs.size == 0:
                continue

            legend_elements.append(
                Line2D([0], [0], marker=markers[p], color='w',
                       label=labels[p], markerfacecolor=colors[p],
                       markersize=8, alpha=0.7))

            ax.scatter(plt_zs, plt_r_es,
                       marker=markers[p], label=labels[p], s=17,
                       color=colors[p], alpha=0.7)

        # Label axes
        ax.set_xlabel(r'$z$')
        ax.set_ylabel('$R_{1/2}/ [pkpc]$')
        ax.set_xlim(4.5, 11.5)
        ax.set_ylim(10 ** -1.5, 10 ** 1.5)

        ax.tick_params(axis='x', which='minor', bottom=True)

        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=4)

        fig.savefig(
            'plots/HalfLightRadius_evolution_' + mtype + '_' + f + '_'
            + orientation + "_" + extinction + "_"
            + '%d.png' % nlim,
            bbox_inches='tight')

        plt.close(fig)
