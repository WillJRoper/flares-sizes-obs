#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import flare.photom as photconv
import h5py
import matplotlib.gridspec as gridspec
from flare import plt as flareplt


# Set plotting fontsizes
plt.rcParams['axes.grid'] = True

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

all_snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
             '010_z005p000']

# Define filter
# filters = ('FAKE.TH.FUV', 'FAKE.TH.NUV', 'FAKE.TH.V')
filters = ['FAKE.TH.' + f
           for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                     'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]

cmap = mpl.cm.get_cmap('coolwarm', len(filters))

reddest_NIRCam = 4.408
reddest_NIRCam_low = 3.880
reddest_NIRCam_high = 4.986

nc_ys = np.logspace(23, 28.5, len(all_snaps))
nc_y_labels = np.logspace(23 + 0.15, 28.5 + 0.15, len(all_snaps))
nc_piv = []
nc_up = []
nc_down = []
nc_zs = []

for snap in all_snaps:
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])
    nc_zs.append(z)
    nc_piv.append(reddest_NIRCam / (1 + z))
    nc_up.append(reddest_NIRCam_high / (1 + z))
    nc_down.append(reddest_NIRCam_low / (1 + z))

nc_piv = np.array(nc_piv)
nc_up = np.array(nc_up)
nc_down = np.array(nc_down)

trans = {}
plt_lams = []
cents = []
bounds = []
lam_max = 0
ls = {}
ts = {}
i = 1
for f in filters:
    l, t = np.loadtxt(filter_path + '/' + '/'.join(f.split('.')) + '.txt',
                      skiprows=1).T
    l /= 10000  # Angstrom to microns
    ls[f] = l
    ts[f] = t
    wid = np.max(l[t > 0]) - np.min(l[t > 0])
    trans[f] = []
    trans[f].append(np.min(l[t > 0]))
    trans[f].append(i)
    trans[f].append(np.max(l[t > 0]))
    plt_lams.append(np.max(l[t > 0]) - (wid / 2))
    cents.append(i)
    bounds.append(i - 0.5)
    print(f.split(".")[-1], np.min(l[t > 0]), np.max(l[t > 0]))

    for snap in all_snaps:
        z_str = snap.split('z')[1].split('p')
        z = float(z_str[0] + '.' + z_str[1])
        if np.min(l[t > 0]) * (1 + z) < reddest_NIRCam_high and np.max(
                l[t > 0]) * (1 + z) > reddest_NIRCam_low:
            print(z, f.split(".")[-1])

    if np.max(l[t > 0]) > lam_max:
        lam_max = np.max(l[t > 0])
    i += 1

bounds.append(i - 0.5)

sinds = np.argsort(plt_lams)
plt_lams = np.array(plt_lams)[sinds]
filters = np.array(filters)[sinds]

filter_labels = [f.split(".")[-1] for f in filters]

bounds = list(sorted(bounds))

norm = cm.Normalize(vmin=min(bounds),
                    vmax=max(bounds),
                    clip=True)

csoft = 0.001802390 / (0.6777) * 1e3

sedint_dict = {}
sedtot_dict = {}
sedlam_dict = {}
imgtot_dict = {}
imgint_dict = {}
mass = []

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['010_z005p000', ]

lim = 5

np.random.seed(100)

int_cmap = mpl.cm.get_cmap('plasma', len(filters))
tot_cmap = mpl.cm.get_cmap('plasma', len(filters))
znorm = cm.Normalize(vmin=5, vmax=10)

fig = plt.figure(figsize=(3.5 * 1.2, 3.5 * 1.4))
gs = gridspec.GridSpec(2, 1, height_ratios=[10, 7])
gs.update(wspace=0.0, hspace=0.0)
ax = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax.semilogy()
# ax.loglog()
# ax1.semilogx()

for i, f in enumerate(filters):
    h = [1, 0.8, 0.6][i % 3]
    ts[f][ts[f] == 1] = h
    ax1.plot(ls[f], ts[f], color=cmap(norm(trans[f][1])))

ax1.tick_params("y", which="both", left=False, right=False,
                labelleft=False, labelright=False)

for snap in snaps:
    for reg in regions:

        f = "FAKE.TH.FUV"

        try:
            hdf = h5py.File(
                "data/flares_sizes_kernelproject_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                   "Total",
                                                                   orientation,
                                                                   f.split(
                                                                       ".")[
                                                                       -1]),
                "r")

            try:
                sedint_dict.setdefault(f, []).extend(
                    hdf[f]["SED_intrinsic"][...])
                sedtot_dict.setdefault(f, []).extend(hdf[f]["SED_total"][...])
                sedlam_dict.setdefault(f, []).extend(
                    hdf[f]["SED_lambda"][...] / 10000)
                mass.extend(hdf[f]["Mass"][...])
            except KeyError as e:
                print(e)
                continue

            hdf.close()

        except OSError:
            continue

    print("Plotting for:")
    print("Snapshot = ", snap)

    legend_elements = []

    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    sedint = np.array(sedint_dict[f])
    sedtot = np.array(sedtot_dict[f])
    sedlam = np.array(sedlam_dict[f])

    if sedint.size == 0:
        continue

    print(f, sedint.shape, np.log10(np.max(mass)))

    # if sedtot.shape[0] > 1:
    #     for i in range(sedtot.shape[0]):
    #         ax.plot(sedlam[i, :], sedtot[i, :],
    #                 color="r", alpha=0.01)
    #         ax.plot(sedlam[i, :], sedint[i, :],
    #                 color="g", alpha=0.01, linestyle="--")
    # else:
    #
    #     ax.plot(sedlam, sedtot,
    #             color="r", alpha=0.01)
    #     ax.plot(sedlam, sedint,
    #             color="g", alpha=0.01)
    print(sedtot.shape)
    print(sedtot)
    ax.plot(sedlam[0, :], np.percentile(sedtot, 50, axis=0),
            color="r", label="Attenuated")
    ax.plot(sedlam[0, :], np.percentile(sedint, 50, axis=0),
            color="g", label="Intrinsic")

    # ywidth = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
    # xwidth = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.1
    #
    # y_low = ax.get_ylim()[1] - ywidth
    # x_low = ax.get_xlim()[0]
    #
    # axin1 = ax.inset_axes([x_low, y_low, xwidth, ywidth],
    #                       transform=ax.transData)
    # axin2 = ax.inset_axes([x_low + xwidth, y_low, xwidth, ywidth],
    #                       transform=ax.transData)

    # for axi in [axin1, axin2]:
    #
    #     axi.grid(False)
    #
    #     # Remove axis labels and ticks
    #     axi.tick_params(axis='x', top=False, bottom=False,
    #                     labeltop=False, labelbottom=False)
    #
    #     axi.tick_params(axis='y', left=False, right=False,
    #                     labelleft=False, labelright=False)
    #
    # axin1.imshow(imgtot[max_ind, :, :], cmap=cmr.cosmic)
    # axin2.imshow(imgint[max_ind, :, :], cmap=cmr.cosmic)

ax.errorbar(nc_piv, nc_ys, xerr=np.array([nc_piv - nc_down, nc_up - nc_piv]),
            capsize=4, color="k", linestyle="none", marker="s", markersize=0)
for ind, s in enumerate(nc_zs):
    ax.annotate(str(int(s)), (nc_piv[ind], nc_y_labels[ind]),
                horizontalalignment="center", fontsize=10)

ax.set_xlim(0.01, 2.1)
ax1.set_xlim(0.01, 2.1)
ax.set_ylim(10 ** 22., 10 ** 33.)
ax1.set_ylim(0.01, 1.15)

ax1.set_xlabel("$\lambda / [\mu\mathrm{m}]$")
ax.set_ylabel("$L / [\mathrm{erg} / \mathrm{s} / \mathrm{Hz}]$")
# ax1.set_ylabel("$T$")

ax.tick_params(axis='y', which='both', left=True)
ax1.tick_params(axis='both', which='both', left=True, bottom=True)

ax.legend()

# create a second axes for the colorbar
print(cents, bounds)
ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                               spacing='uniform', ticks=cents,
                               boundaries=bounds, format='%1i')
cb.set_ticklabels(filter_labels)
cb.ax.tick_params(axis='both', which="minor", left=False, right=False)

string = 'plots/SED/SED' + "_" + orientation + "_" + extinction
fig.savefig(string.replace(".", "p") + ".pdf",
            bbox_inches='tight', dpi=100)
plt.close(fig)
