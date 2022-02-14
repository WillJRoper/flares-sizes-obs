#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib as mpl
from matplotlib.lines import Line2D
from astropy.cosmology import Planck13 as cosmo
from flare.photom import M_to_lum
import flare.photom as photconv
import pandas as pd
import weighted
from matplotlib.cbook import violin_stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
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


fit = lambda z, C, m: C * (1 + z) ** -m


def norm_fit(z, m):
    return (1 + z) ** -m


# Bouwens et al. (2004, 2006) claim that the relation is roughly (1 + z)−1,
# suggestive that the sizes of disks scale with constant halo mass,
# while Ferguson et al. (2004) and Hathi et al. (2008) argue that (1  +  z)−1.5

oesch_up_m = (1.12, 0.17, 0.17)  # (0.3-1)L*
hol_up_m = (1.3, 0.4, 0.4)  # L<0.3L*
oesch_low_m = (1.32, 0.52, 0.52)  # L<0.3L*
hol_low_m = (0.76, 0.12, 0.12)  # 0.3L*<L
bt_up_m = (0.662, 0.008, 0.008)  # (0.3-1)L*
bouwens_m = (1.05, 0.21,
             0.21)  # (0.3-1)L* https://iopscience.iop.org/article/10.1086/423786/pdf
ono_low_m = (1.3, 0.12,
             0.14)  # (0.3-1)L* https://iopscience.iop.org/article/10.1088/0004-637X/777/2/155
ono_up_m = (1.3, 0.12,
            0.14)  # L<0.3L* https://iopscience.iop.org/article/10.1088/0004-637X/777/2/155
ono_up_norm = (1., 0.09,
               0.07)  # (0.3-1)L* https://iopscience.iop.org/article/10.1088/0004-637X/777/2/155
ono_low_norm = (0.88, 0.08,
                0.09)  # L<0.3L* https://iopscience.iop.org/article/10.1088/0004-637X/777/2/155
kawa_up_norm = (1.28, 0.11,
                0.11)  # (0.3-1)L* https://iopscience.iop.org/article/10.3847/1538-4357/aaa6cf

# Define Kawamata17 fit and parameters
kawa_params = {'beta': {6: 0.46, 7: 0.46, 8: 0.38, 9: 0.56},
               'r_0': {6: 0.94, 7: 0.94, 8: 0.81, 9: 1.2}}
kawa_fit = lambda l, r0, b: r0 * (l / M_to_lum(-21)) ** b
ono_fit = lambda z, a, m: 10 ** (m * np.log10(1 + z) + a)

L_star = M_to_lum(-21)


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


def size_evo_violin(data, intr_data, snaps, f, mtype, orientation, Type,
                    extinction):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Filter =", f)

    hlr = []
    hdr = []
    ws = []
    plt_z = []
    ms = []
    lums = []

    for snap in snaps:

        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])

        if mtype == "part":
            hlrs = np.array(data[snap][f]["HLR_0.5"])
            intr_hlrs = np.array(intr_data[snap][f]["HLR_0.5"])
            lumins = np.array(data[snap][f]["Luminosity"])
            intr_lumins = np.array(intr_data[snap][f]["Luminosity"])
        elif mtype == "app":
            hlrs = np.array(data[snap][f]["HLR_Aperture_0.5"])
            intr_hlrs = np.array(intr_data[snap][f]["HLR_Aperture_0.5"])
            lumins = np.array(data[snap][f]["Image_Luminosity"])
            intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])
        else:
            hlrs = np.array(data[snap][f]["HLR_Pixel_0.5"])
            intr_hlrs = np.array(intr_data[snap][f]["HLR_Pixel_0.5"])
            lumins = np.array(data[snap][f]["Image_Luminosity"])
            intr_lumins = np.array(intr_data[snap][f]["Image_Luminosity"])
        m = np.array(data[snap][f]["Mass"])
        hdrs = np.array(data[snap][f]["HDR"])
        w = np.array(data[snap][f]["Weight"])

        compact_com = data[snap][f]["Compact_Population_Complete"]
        diffuse_com = data[snap][f]["Diffuse_Population_Complete"]

        if Type == "NonComplete":
            complete = data[snap][f]["okinds"]
        else:
            complete = np.logical_or(compact_com, diffuse_com)

        if len(w[complete]) == 0:
            continue

        hlr.append(hlrs[complete])
        hdr.append(hdrs[complete])
        ws.append(w[complete])
        ms.append(m[complete])
        lums.append(lumins[complete])

        plt_z.append(z)

    fitting_lums = []
    fitting_hlrs = []
    fitting_hdrs = []
    fitting_zs = []
    fitting_ws = []
    fitting_ms = []

    for i in range(len(hlr)):
        fitting_zs.extend(np.full(len(hlr[i]), plt_z[i]))
        fitting_hlrs.extend(hlr[i])
        fitting_hdrs.extend(hdr[i])
        fitting_ws.extend(ws[i])
        fitting_ms.extend(ms[i])
        fitting_lums.extend(lums[i])

    fitting_hlrs = np.array(fitting_hlrs)
    fitting_zs = np.array(fitting_zs)
    fitting_ws = np.array(fitting_ws)
    fitting_lums = np.array(fitting_lums)

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

    slopes = []
    slopes_low = []
    slopes_high = []
    slope_errors = []
    slope_errors_low = []
    slope_errors_high = []

    for ls, bin, col in zip(["-", "-", "-"], ["low", "mid", "high"],
                            ["b", "g", "r"]):

        print("Bin:", bin)

        if bin == "high":
            okinds = fitting_lums >= 0.3 * L_star
        elif bin == "mid":
            okinds = np.logical_and(fitting_lums >= 0.3 * L_star,
                                    fitting_lums <= L_star)
        else:
            okinds = fitting_lums < 0.3 * L_star

        if fitting_zs[okinds].size == 0:
            slopes.append(np.nan)
            slope_errors.append(np.nan)
            continue

        uni_z = np.unique(fitting_zs[okinds])

        print("Redshifts in sample:", uni_z)

        fit_plt_zs = np.linspace(np.max(uni_z), np.min(uni_z), 1000)

        lowz_okinds = fitting_zs[okinds] <= 10
        highz_okinds = fitting_zs[okinds] >= 7

        popt, pcov = curve_fit(fit, fitting_zs[okinds], fitting_hlrs[okinds],
                               p0=(1, 0.5), sigma=fitting_ws[okinds])

        slopes.append(popt[1])
        slope_errors.append(np.sqrt(pcov[1, 1]))

        print("--------------", "Total", Type, bin,
              mtype, f, "--------------")
        print("C=%.3f +/- %.3f " % (popt[0], np.sqrt(pcov[0, 0])))
        print("m=%.3f +/- %.3f " % (popt[1], np.sqrt(pcov[1, 1])))
        print("Points fit on:", fitting_zs[okinds].size)
        print("----------------------------------------------------------")

        ax.plot(fit_plt_zs, fit(fit_plt_zs, popt[0], popt[1]),
                linestyle="-", color=col)
        
        popt, pcov = curve_fit(fit, fitting_zs[okinds][lowz_okinds], 
                               fitting_hlrs[okinds][lowz_okinds],
                               p0=(1, 0.5), 
                               sigma=fitting_ws[okinds][lowz_okinds])

        slopes_low.append(popt[1])
        slope_errors_low.append(np.sqrt(pcov[1, 1]))

        print("--------------", "Total-Low", Type, bin,
              mtype, f, "--------------")
        print("C=%.3f +/- %.3f " % (popt[0], np.sqrt(pcov[0, 0])))
        print("m=%.3f +/- %.3f " % (popt[1], np.sqrt(pcov[1, 1])))
        print("Points fit on:", fitting_zs[okinds].size)
        print("----------------------------------------------------------")

        fit_plt_zs = np.linspace(np.max(fitting_zs[okinds][lowz_okinds]), 
                                 np.min(fitting_zs[okinds][lowz_okinds]), 1000)

        ax.plot(fit_plt_zs, fit(fit_plt_zs, popt[0], popt[1]),
                linestyle="--", color=col, zorder=3)
        
        popt, pcov = curve_fit(fit, fitting_zs[okinds][highz_okinds], 
                               fitting_hlrs[okinds][highz_okinds],
                               p0=(1, 0.5), 
                               sigma=fitting_ws[okinds][highz_okinds])

        slopes_high.append(popt[1])
        slope_errors_high.append(np.sqrt(pcov[1, 1]))

        print("--------------", "Total-High", Type, bin,
              mtype, f, "--------------")
        print("C=%.3f +/- %.3f " % (popt[0], np.sqrt(pcov[0, 0])))
        print("m=%.3f +/- %.3f " % (popt[1], np.sqrt(pcov[1, 1])))
        print("Points fit on:", fitting_zs[okinds].size)
        print("----------------------------------------------------------")

        fit_plt_zs = np.linspace(np.max(fitting_zs[okinds][highz_okinds]), 
                                 np.min(fitting_zs[okinds][highz_okinds]), 1000)

        ax.plot(fit_plt_zs, fit(fit_plt_zs, popt[0], popt[1]),
                linestyle="dotted", color=col)

        hlr_16 = []
        hlr_84 = []
        med = []
        cnts = []
        for i in range(len(plt_z)):

            zokinds = np.logical_and(fitting_zs[okinds] > plt_z[i] - 0.5,
                                     fitting_zs[okinds] <= plt_z[i] + 0.5)
            cnts.append(fitting_hlrs[okinds][zokinds].size)
            med.append(np.median(fitting_hlrs[okinds][zokinds]))

            if fitting_hlrs[okinds][zokinds].size == 0:
                hlr_16.append(np.nan)
                hlr_84.append(np.nan)
                continue

            hlr_16.append(np.percentile(fitting_hlrs[okinds][zokinds], 16))
            hlr_84.append(np.percentile(fitting_hlrs[okinds][zokinds], 84))

        cnts = np.array(cnts)
        med = np.array(med)
        hlr_16 = np.array(hlr_16)
        hlr_84 = np.array(hlr_84)
        plt_z = np.array(plt_z)

        cnt_okinds = cnts > 10
        ax.errorbar(plt_z[cnt_okinds], med[cnt_okinds],
                    yerr=(hlr_16[cnt_okinds], hlr_84[cnt_okinds]),
                    capsize=5, color=col,
                    marker="s", linestyle="none")
        ax.errorbar(plt_z[~cnt_okinds], med[~cnt_okinds],
                    yerr=(hlr_16[~cnt_okinds], hlr_84[~cnt_okinds]),
                    capsize=5, color=col,
                    marker="^", linestyle="none", alpha=0.6)

    legend_elements.append(Line2D([0], [0], color='b',
                                  label="$L < 0.3 L^{\star}_{z=3}$",
                                  linestyle="-"))

    legend_elements.append(Line2D([0], [0], color='g',
                                  label="$0.3L^{\star}_{z=3}\leq L "
                                        "\leq L^{\star}_{z=3}$",
                                  linestyle="-"))

    legend_elements.append(Line2D([0], [0], color='r',
                                  label="$0.3 L^{\star}_{z=3} \leq L$",
                                  linestyle="-"))

    legend_elements.append(Line2D([0], [0], color='k',
                                  label="FLARES ($5\leq z \leq 12$)",
                                  linestyle="-"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="FLARES ($5\leq z \leq 10$)",
                                  linestyle="--"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="FLARES ($7\leq z \leq 12$)",
                                  linestyle="dotted"))

    legend_elements.append(Line2D([0], [0], color='k',
                                  label="Simulations",
                                  linestyle="none", marker="s"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="Simulations ($N<10$)",
                                  linestyle="none", marker="^"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="Observations",
                                  linestyle="none", marker="*"))

    # Label axes
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('$R/ [\mathrm{pkpc}]$')

    ax.tick_params(axis='both', which='both', left=True, bottom=True)

    ax.set_xlim(4.77, 12.5)
    ax.set_ylim(10 ** -1.6, 10 ** 0.8)

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3)

    fig.savefig(
        'plots/Violin_ObsCompHalfLightRadius_evolution_' + mtype + '_' + f + '_'
        + orientation + "_" + extinction + "_" + Type + ".pdf",
        bbox_inches='tight')

    plt.close(fig)

    fig = plt.figure()
    bar_ax = fig.add_subplot(111)

    # Plot FIRE region
    bar_ax.fill_between([-2, 30], y1=[1, 1], y2=[2, 2],
                        facecolor="k", alpha=0.2)

    comp_val_works = ["FLARES", "FLARES (low-z)", "FLARES (high-z)",
                      "Marshall+", "Oesch+",
                      "Holwerda+", "Kawamata+", "Ono+"]
    comp_vals = [(slopes, slope_errors, []),
                 (slopes_low, slope_errors_low, []),
                 (slopes_high, slope_errors_high, []),
                 ([], bt_up_m, []), 
                 (oesch_low_m, oesch_up_m, []), 
                 (hol_low_m, [], hol_up_m), 
                 ([], kawa_up_norm, []), 
                 (ono_low_m, ono_up_m, [])]
    x = -1
    for (i, w), v_lst in zip(enumerate(comp_val_works), comp_vals):
        for c, j in zip(["b", "g", "r"], range(3)):
            x += 1

            if w[0] == "F" :
                print(w, v_lst, v_lst[0], v_lst[0][j])
                bar_ax.errorbar([x, ],
                                [v_lst[0][j], ],
                                yerr=np.array([(v_lst[1][j], v_lst[1][j])]).T,
                                color=c, fmt="s", capsize=5)

            elif w[0] == "M":

                bar_ax.errorbar([x, ],
                                [v_lst[j][0], ],
                                yerr=np.array([(v_lst[j][2], v_lst[j][1])]).T,
                                color=c, fmt="s", capsize=5)
            else:
                if len(v_lst[j]) == 0:
                    continue

                bar_ax.errorbar([x, ],
                                [v_lst[j][0], ],
                                yerr=np.array([(v_lst[j][2], v_lst[j][1])]).T,
                                color=c, fmt="*", capsize=5, markersize=7)


    # bar_ax.axvline(2.5, linestyle="-", linewidth=1, color="grey", alpha=0.3)
    # bar_ax.axvline(5.5, linestyle="-", linewidth=1, color="grey", alpha=0.3)
    # bar_ax.axvline(8.5, linestyle="-", linewidth=1, color="grey", alpha=0.3)
    # bar_ax.axvline(11.5, linestyle="-", linewidth=1, color="grey", alpha=0.3)
    # bar_ax.axvline(14.5, linestyle="-", linewidth=1, color="grey", alpha=0.3)

    bar_ax.axhline(1, linestyle="--", color="grey", alpha=0.7)
    bar_ax.axhline(1.5, linestyle="dotted", color="grey", alpha=0.7)

    bar_ax.set_xlim(-0.5, 23.5)
    bar_ax.set_ylim(0.49, 3)

    bar_ax.tick_params(reset=True, bottom=True, left=True,
                       top=False, right=False)

    for pos in [-0.5, 2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5, 23.5]:
        bar_ax.axvline(pos, linestyle="-", color="k", alpha=0.5)

    bar_ax.set_ylabel("$m$")
    bar_ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22])
    bar_ax.set_xticklabels(comp_val_works, rotation=90)
    bar_ax.tick_params(axis='x', which='minor', bottom=True)

    # bar_ax.grid(False)
    # bar_ax.grid(axis="y", linestyle="-", linewidth=1, color="grey", alpha=0.3)

    fig.savefig(
        'plots/SlopeComp_HalfLightRadius_evolution_' + mtype + '_' + f + '_'
        + orientation + "_" + extinction + "_" + Type + ".pdf",
        bbox_inches='tight')

    plt.close(fig)

