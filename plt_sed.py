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
import matplotlib as mpl
import flare.photom as photconv
import h5py

sns.set_context("paper")
sns.set_style('whitegrid')

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

cmap = mpl.cm.get_cmap('viridis', len(filters))

reddest_NIRCam = 4.408
reddest_NIRCam_low = 3.880
reddest_NIRCam_high = 4.986

nc_ys = np.logspace(22, 26.5, len(all_snaps))
nc_y_labels = np.logspace(22 + 0.15, 26.5 + 0.15, len(all_snaps))
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
i = 1
for f in filters:
    l, t = np.loadtxt(filter_path + '/' + '/'.join(f.split('.')) + '.txt',
                      skiprows=1).T
    l /= 10000  # Angstrom to microns
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

bounds.append(i + 0.5)

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
mass_dict = {}

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

fig = plt.figure(figsize=(6.4, 3.8))
ax = fig.add_subplot(111)
ax.loglog()

for f in filters:
    ax.axvspan(trans[f][0], trans[f][2],
               color=cmap(norm(trans[f][1])))

for snap in snaps:
    for reg in regions:

        f = "FAKE.TH.FUV"

        try:
            hdf = h5py.File(
                "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Total",
                                                            orientation),
                "r")

            try:
                sedint_dict.setdefault(f, []).extend(
                    hdf[f]["SED_intrinsic"][...])
                sedtot_dict.setdefault(f, []).extend(hdf[f]["SED_total"][...])
                sedlam_dict.setdefault(f, []).extend(
                    hdf[f]["SED_lambda"][...] / 10000)
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

    print(sedint.shape)

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
            capsize=5, color="k", linestyle="none", marker="s", markersize=0)
for ind, s in enumerate(nc_zs):
    ax.annotate(str(int(s)), (nc_piv[ind], nc_y_labels[ind]),
                horizontalalignment="center", fontsize=6)

ax.set_xlim(0.05, None)
ax.set_ylim(10 ** 20., 10 ** 33.)

ax.set_xlabel("$\lambda / [\mu\mathrm{m}]$")
ax.set_ylabel("$L_{" + f.split(".")[-1]
              + r"} / [\mathrm{erg} / \mathrm{s} / \mathrm{Hz}]$")

ax.legend()

# create a second axes for the colorbar
ax2 = fig.add_axes([0.95, 0.1, 0.015, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                               spacing='uniform', ticks=cents,
                               boundaries=bounds, format='%1i')
cb.set_ticklabels(filter_labels)

string = 'plots/SED/SED' + "_" + orientation + "_" + extinction
fig.savefig(string.replace(".", "p") + ".png",
            bbox_inches='tight', dpi=100)
plt.close(fig)
