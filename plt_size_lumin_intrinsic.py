#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

mpl.use('Agg')
warnings.filterwarnings('ignore')
import flare.photom as photconv
from flare import plt as flareplt

# Set plotting fontsizes
plt.rcParams['axes.grid'] = True
flareplt.rcParams['axes.grid'] = True

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE


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


def size_lumin_intrinsic(hlrs, lumins, w, com_comp, diff_comp, com_ncomp,
                         diff_ncomp, f, snap,
                         mtype, orientation, Type, extinction, weight_norm,
                         extent):
    print("Plotting for:")
    print("Orientation =", orientation)
    print("Type =", Type)
    print("Filter =", f)
    print("Snap =", snap)

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    if w.size <= 1:
        return

    fig = plt.figure(figsize=(1.2 * 3.5, 2.2 * 3.5))
    gs = gridspec.GridSpec(2, 4, height_ratios=(3, 10),
                           width_ratios=(20, 3, 1, 1))
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 2])
    cax2 = fig.add_subplot(gs[1, 3])
    axtop = fig.add_subplot(gs[0, 0])
    axright = fig.add_subplot(gs[1, 1:])
    axtop.loglog()
    axright.loglog()
    axtop.grid(False)
    axright.grid(False)
    try:
        cbar = ax.hexbin(lumins[diff_comp], hlrs[diff_comp],
                          C=w[diff_comp], gridsize=50,
                          mincnt=np.min(w) - (0.1 * np.min(w)),
                          xscale='log', yscale='log',
                          norm=weight_norm, linewidths=0.2,
                          cmap='Greys',
                          extent=[extent[2], extent[3], extent[0],
                                  extent[1]])
        cbar = ax.hexbin(lumins[com_comp], hlrs[com_comp],
                         C=w[com_comp], gridsize=50,
                         mincnt=np.min(w) - (0.1 * np.min(w)),
                         xscale='log', yscale='log',
                         norm=weight_norm, linewidths=0.2,
                         cmap='viridis', extent=[extent[2], extent[3],
                                                 extent[0], extent[1]])
    except ValueError as e:
        print(e)

    lumin_bins = np.logspace(extent[2], extent[3], 50)
    Hbot_all, bin_edges = np.histogram(lumins, bins=lumin_bins, weights=w)
    Hbot_com, bin_edges = np.histogram(lumins[com_comp], bins=lumin_bins,
                                       weights=w[com_comp])
    Hbot_diff, bin_edges = np.histogram(lumins[diff_comp], bins=lumin_bins,
                                        weights=w[diff_comp])
    lbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axtop.plot(lbin_cents, Hbot_com, color="g")
    axtop.plot(lbin_cents, Hbot_diff, color="k")
    axtop.plot(lbin_cents, Hbot_all, color="r", linestyle="--")

    hmr_bins = np.logspace(extent[0], extent[1], 50)
    Htop_all, bin_edges = np.histogram(hlrs, bins=hmr_bins, weights=w)
    Htop_com, bin_edges = np.histogram(hlrs[com_comp], bins=hmr_bins,
                                       weights=w[com_comp])
    Htop_diff, bin_edges = np.histogram(hlrs[diff_comp], bins=hmr_bins,
                                        weights=w[diff_comp])
    hmrbin_cents = (bin_edges[1:] + bin_edges[:-1]) / 2

    axright.plot(Htop_all, hmrbin_cents, color="r", linestyle="--",
                 label="All", zorder=2)
    axright.plot(Htop_com, hmrbin_cents, color="g", label="Compact", zorder=1)
    axright.plot(Htop_diff, hmrbin_cents, color="k", label="Diffuse", zorder=0)

    # Remove axis labels and ticks
    axtop.tick_params(axis='x', top=False, bottom=False,
                      labeltop=False, labelbottom=False)
    axright.tick_params(axis='y', left=False, right=False,
                        labelleft=False, labelright=False)
    cax.tick_params(axis='x', top=False, bottom=False,
                     labeltop=False, labelbottom=False)
    cax.tick_params(axis='y', left=False, right=False,
                     labelleft=False, labelright=False)
    cax2.tick_params(axis='x', top=False, bottom=False,
                     labeltop=False, labelbottom=False)
    cax2.tick_params(axis='y', left=False, right=False,
                     labelleft=False, labelright=False)

    axtop.spines['top'].set_visible(False)
    axtop.spines['right'].set_visible(False)
    axright.spines['top'].set_visible(False)
    axright.spines['right'].set_visible(False)

    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("Greys"),
                                    norm=weight_norm)
    cb1.ax.yaxis.set_ticks([])
    cb1 = mpl.colorbar.ColorbarBase(cax2, cmap=plt.get_cmap("viridis"),
                                    norm=weight_norm)
    cb1.set_label("$\sum w_{i}$")

    # Label axes
    ax.set_xlabel(r"$L_{\mathrm{" + f.split(".")[-1]
                   + "}}/$ [erg $/$ s $/$ Hz]")
    ax.set_ylabel('$R/ [pkpc]$')
    ax.tick_params(axis='both', which='both', left=True, bottom=True)
    axtop.set_ylabel("$N$")
    axright.set_xlabel("$N$")

    axtop.tick_params(axis='y', which='both', left=True)
    axright.tick_params(axis='x', which='both', bottom=True)

    ax.set_xlim(10 ** extent[2], 10 ** extent[3])
    ax.set_ylim(10 ** extent[0], 10 ** extent[1])
    axtop.set_xlim(10 ** extent[2], 10 ** extent[3])
    axright.set_ylim(10 ** extent[0], 10 ** extent[1])

    handles, labels = axright.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best")

    fig.savefig(
        'plots/' + str(z) + '/HalfLightRadius_' + mtype + "_" + f + '_' + str(
            z) + '_'
        + orientation + "_Intrinsic_" + extinction + ".pdf",
        bbox_inches='tight')

    plt.close(fig)
