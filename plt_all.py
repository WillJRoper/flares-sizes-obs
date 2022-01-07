import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from plt_half_dust_radius_comp import hdr_comp
from plt_mass_lumin import mass_lumin
from plt_no_smooth_size_lumin_fitgrid import fit_size_lumin_grid_nosmooth
from plt_size_evo_violin import size_evo_violin
from plt_size_lumin_fitgrid import fit_size_lumin_grid
from plt_size_lumin_grid import size_lumin_grid
from plt_size_lumin_grid_all_filters import size_lumin_grid_allf
from plt_size_lumin_intrinsic import size_lumin_intrinsic
from plt_size_part_smooth_comp import size_comp_smooth_part
from plt_size_smooth_comp import size_comp_smooth
from plt_size_type_comp import size_comp
from plt_img_type_comp import img_size_comp

# Set orientation
orientation = "sim"

snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
         '010_z005p000']
all_snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000']  # used for fitting
total_snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
               '003_z012p000', '004_z011p000', '005_z010p000',
               '006_z009p000', '007_z008p000', '008_z007p000',
               '009_z006p000', '010_z005p000']  # every output
limed_snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
               '006_z009p000', '007_z008p000', '008_z007p000']
low_limed_snaps = ['005_z010p000', '006_z009p000', '007_z008p000',
                   '008_z007p000', '009_z006p000', '010_z005p000']

# Define filter
all_filters = ['FAKE.TH.' + f
               for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
                         'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
# filters = ['FAKE.TH.' + f for f in ['FUV', 'MUV', 'NUV']]
filters = ['FAKE.TH.' + f for f in ['FUV', ]]

keys = ["Mass", "Image_Luminosity", "HLR_0.5",
        "HLR_Pixel_0.5", "Luminosity",
        "HDR", "nStar", "HLR_Pixel_0.5_No_Smooth",
        "Image_Luminosity_No_Smooth"]

csoft = 0.001802390 / (0.6777) * 1e3

data = {}
intr_data = {}

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

    for snap in all_snaps:
        reg_snaps.append((reg, snap))

# Define dictionary for image size
img_shapes = {}

for reg, snap in reg_snaps:

    for f in all_filters:

        try:
            hdf = h5py.File(
                "data/flares_sizes_all_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                   "Total",
                                                                   orientation,
                                                                   f.split(
                                                                       ".")[
                                                                       -1]),
                "r")
        except OSError as e:
            print(reg, snap, e)
            continue

        data.setdefault(snap, {})
        intr_data.setdefault(snap, {})

        data[snap].setdefault(f, {})
        intr_data[snap].setdefault(f, {})

        for key in keys:
            try:
                data[snap][f].setdefault(key, []).extend(hdf[f][key][...])
            except KeyError as e:
                print(reg, snap, e)

        img_shape = hdf[f]["Images"].shape
        if len(img_shape) > 2:
            img_shapes[snap] = (img_shape[1],
                                img_shape[2])

        data[snap][f].setdefault("Weight", []).extend(
            np.full(hdf[f]["Mass"][...].size, weights[int(reg)]))

        hdf.close()

        try:
            hdf = h5py.File(
                "data/flares_sizes_all_{}_{}_{}_{}_{}.hdf5".format(reg, snap,
                                                                   "Intrinsic",
                                                                   orientation,
                                                                   f.split(
                                                                       ".")[
                                                                       -1]),
                "r")
        except OSError as e:
            print(reg, snap, e)
            continue

        surf_dens = hdf[f]["Luminosity"][...] \
                    / (np.pi * (2 * hdf[f]["HLR_0.5"][...]) ** 2)

        intr_data[snap][f].setdefault("Inner_Surface_Density",
                                      []).extend(surf_dens)

        intr_data[snap][f].setdefault("Weight", []).extend(
            np.full(hdf[f]["Mass"][...].size, weights[int(reg)]))

        for key in keys:
            try:
                intr_data[snap][f].setdefault(key, []).extend(hdf[f][key][...])
            except KeyError as e:
                print(reg, snap, e)

        hdf.close()

# Count the number of galaxies in FLARES
all_n_gals = 0
all_n_gals_above_100 = 0
all_comp_gals = 0
all_diffuse_gals = 0
for snap in intr_data.keys():
    okinds = np.array(intr_data[snap][filters[0]]["Mass"]) > 10 ** 8
    n_gals = np.array(intr_data[snap][filters[0]]["Mass"])[okinds].size
    all_n_gals += n_gals
    okinds = np.array(intr_data[snap][filters[0]]["nStar"]) > 100
    n_gals_above_100 = np.array(intr_data[snap][filters[0]]["Mass"])[
        okinds].size
    all_n_gals_above_100 += n_gals_above_100

    okinds = np.array(intr_data[snap][filters[0]]["nStar"]) > 100
    sample = np.array(intr_data[snap][filters[0]][
                          "Inner_Surface_Density"])[okinds]
    sfd_okinds = sample >= 10 ** 29
    comp_gals = sample[sfd_okinds].size
    diffuse_gals = sample[~sfd_okinds].size

    all_comp_gals += comp_gals
    all_diffuse_gals += diffuse_gals

    print("Galaxies with M_star/M_sun>10**8 in snapshot %s: %d"
          % (snap, n_gals))
    print("Galaxies with N_star>100 in snapshot %s: %d"
          % (snap, n_gals_above_100))
    print("Compact galaxies in snapshot %s: %d"
          % (snap, comp_gals))
    print("Diffuse galaxies in snapshot %s: %d"
          % (snap, diffuse_gals))

print("Total galaxies with M_star/M_sun>10**8: %d" % all_n_gals)
print("Total galaxies with N_star>100: %d" % all_n_gals_above_100)
print("Total compact galaxies: %d" % all_comp_gals)
print("Total diffuse galaxies: %d" % all_diffuse_gals)

print("Image Dimensions:")
print(img_shapes)

for snap in all_snaps:

    for f in all_filters:
        okinds = np.ones(len(intr_data[snap][f]["nStar"]), dtype=bool)
        for key in keys:
            okinds = np.logical_and(okinds, np.logical_and(
                np.array(data[snap][f][key]) > 0,
                np.array(intr_data[snap][f][key]) > 0))

        for key in data[snap][f].keys():
            data[snap][f][key] = np.array(data[snap][f][key])[okinds]

        for key in intr_data[snap][f].keys():
            intr_data[snap][f][key] = np.array(intr_data[snap][f][key])[okinds]

        okinds = intr_data[snap][f]["nStar"] >= 100

        data[snap][f]["okinds"] = okinds
        intr_data[snap][f]["okinds"] = okinds

        data[snap][f]["Complete_Luminosity"] = np.percentile(
            data[snap][f]["Image_Luminosity"][~okinds], 95)
        data[snap][f]["Complete_Mass"] = np.percentile(
            data[snap][f]["Mass"][~okinds], 95)
        intr_data[snap][f]["Complete_Luminosity"] = np.percentile(
            intr_data[snap][f]["Image_Luminosity"][~okinds], 95)
        intr_data[snap][f]["Complete_Mass"] = np.percentile(
            intr_data[snap][f]["Mass"][~okinds], 95)

        # data[snap][f]["Complete_Luminosity"] = np.max(
        #     data[snap][f]["Image_Luminosity"][~okinds])
        # data[snap][f]["Complete_Mass"] = np.max(
        #     data[snap][f]["Mass"][~okinds])
        # intr_data[snap][f]["Complete_Luminosity"] = np.max(
        #     intr_data[snap][f]["Image_Luminosity"][~okinds])
        # intr_data[snap][f]["Complete_Mass"] = np.max(
        #     intr_data[snap][f]["Mass"][~okinds])

        print("Intrinsic: complete luminosity/mass for", snap, f,
              "%.2f/%.2f" % (
                  np.log10(intr_data[snap][f]["Complete_Luminosity"]),
                  np.log10(intr_data[snap][f]["Complete_Mass"])))
        print("Total: complete luminosity/mass for", snap, f,
              "%.2f/%.2f" % (np.log10(data[snap][f]["Complete_Luminosity"]),
                             np.log10(data[snap][f]["Complete_Mass"])))
        print("----------------------------------------------------------")

        okinds = np.logical_and(
            data[snap][f]["Image_Luminosity"] >= data[snap][f][
                "Complete_Luminosity"], data[snap][f]["Mass"] >= data[snap][f][
                "Complete_Mass"])
        intr_okinds = np.logical_and(
            intr_data[snap][f]["Image_Luminosity"] >= intr_data[snap][f][
                "Complete_Luminosity"],
            intr_data[snap][f]["Mass"] >= intr_data[snap][f]["Complete_Mass"])

        compact_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) >= 10 ** 29
        diffuse_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) < 10 ** 29

        data[snap][f]["Compact_Population"] = compact_pop
        data[snap][f]["Diffuse_Population"] = diffuse_pop
        intr_data[snap][f]["Compact_Population"] = compact_pop
        intr_data[snap][f]["Diffuse_Population"] = diffuse_pop

        data[snap][f]["Compact_Population_Complete"] = np.logical_and(
            compact_pop, okinds)
        data[snap][f]["Diffuse_Population_Complete"] = np.logical_and(
            diffuse_pop, okinds)
        intr_data[snap][f]["Compact_Population_Complete"] = np.logical_and(
            compact_pop, intr_okinds)
        intr_data[snap][f]["Diffuse_Population_Complete"] = np.logical_and(
            diffuse_pop, intr_okinds)

        data[snap][f]["Compact_Population_NotComplete"] = np.logical_and(
            compact_pop, ~okinds)
        data[snap][f]["Diffuse_Population_NotComplete"] = np.logical_and(
            diffuse_pop, ~okinds)
        intr_data[snap][f]["Compact_Population_NotComplete"] = np.logical_and(
            compact_pop, ~intr_okinds)
        intr_data[snap][f]["Diffuse_Population_NotComplete"] = np.logical_and(
            diffuse_pop, ~intr_okinds)

# Count the number of galaxies in FLARES
all_complete_gals = 0
for snap in intr_data.keys():
    okinds = np.logical_or(
        data[snap][filters[0]]["Compact_Population_Complete"],
        data[snap][filters[0]]["Diffuse_Population_Complete"])
    complete_gals = data[snap][filters[0]]["Mass"][okinds].size
    all_complete_gals += complete_gals

    print("Complete Galaxies in snapshot %s: %d"
          % (snap, complete_gals))

print("Total Complete galaxies: %d" % all_complete_gals)

# Define the norm
weight_norm = LogNorm(vmin=10 ** -4, vmax=1)

# Define size luminosity limits
xlims = (10 ** 27.2, 10 ** 30.7)
ylims = (10 ** -1.1, 10 ** 1.1)

print("--------------------------- Size-Lumin ---------------------------")
size_lumin_grid(data, snaps, filters, orientation, "Total",
                "default", "pix", weight_norm, xlims, ylims)
# size_lumin_grid(data, snaps, filters, orientation, "Total",
#                 "default", "app", weight_norm)
print("--------------------------- Fits ---------------------------")
fit_size_lumin_grid(data, intr_data, snaps, filters, orientation, "Total",
                    "default",
                    "pix", "Complete", xlims, ylims)
print("--------------------------- Fits No Smooth ---------------------------")
fit_size_lumin_grid_nosmooth(data, snaps, filters, orientation, "Total",
                             "default",
                             "pix", "Complete", xlims, ylims, weight_norm)
print(
    "--------------------------- Fits Incomplete ---------------------------")
fit_size_lumin_grid(data, intr_data, snaps, filters, orientation, "Total",
                    "default",
                    "pix", "All", xlims, ylims)
# fit_size_lumin_grid(data, intr_data, snaps, filters, orientation, "Total",
#                     "default",
#                     "app")
print("--------------------------- All filters ---------------------------")
size_lumin_grid_allf(data, intr_data, snaps, all_filters, orientation,
                     "Total", "default",
                     "pix", weight_norm, list(xlims), list(ylims), "Complete")
print(
    "--------------------------- All filters Incomplete ---------------------------")
size_lumin_grid_allf(data, intr_data, snaps, all_filters, orientation,
                     "Total", "default",
                     "pix", weight_norm, list(xlims), list(ylims), "All")

for f in filters:
    print(f)
    print("--------------------------- Evolution ---------------------------")
    size_evo_violin(data, intr_data, all_snaps, f, "pix", "sim", "All",
                    "default")
    print(
        "--------------------------- Evolution Incomplete ---------------------------")
    size_evo_violin(data, intr_data, all_snaps, f, "pix", "sim", "NonComplete",
                    "default")
    print(
        "--------------------------- Evolution Incomplete Particle ---------------------------")
    size_evo_violin(data, intr_data, all_snaps, f, "part", "sim", "All",
                    "default")
    # size_evo_violin(data, intr_data, all_snaps, f, "app", "sim", "All",
    #                 "default")
    print(
        "--------------------------- Evolution Limited to high ---------------------------")
    size_evo_violin(data, intr_data, limed_snaps, f, "pix", "sim", "Limited",
                    "default")
    print(
        "--------------------------- Evolution Limited to low ---------------------------")
    size_evo_violin(data, intr_data, low_limed_snaps, f, "pix", "sim",
                    "Low-Limited",
                    "default")
    for snap in snaps:
        print(snap)
        print(
            "--------------------------- Mass Lumin", snap,
            "---------------------------")
        mass_lumin(intr_data[snap][f]["Mass"],
                   intr_data[snap][f]["Image_Luminosity"],
                   intr_data[snap][f]["Compact_Population_Complete"],
                   intr_data[snap][f]["Diffuse_Population_Complete"],
                   intr_data[snap][f]["Compact_Population_NotComplete"],
                   intr_data[snap][f]["Diffuse_Population_NotComplete"],
                   intr_data[snap][f]["okinds"],
                   data[snap][f]["Weight"],
                   f, snap, orientation, "Intrinsic", "default",
                   intr_data[snap][f]["Complete_Luminosity"],
                   intr_data[snap][f]["Complete_Mass"], weight_norm)
        mass_lumin(data[snap][f]["Mass"],
                   data[snap][f]["Image_Luminosity"],
                   data[snap][f]["Compact_Population_Complete"],
                   data[snap][f]["Diffuse_Population_Complete"],
                   data[snap][f]["Compact_Population_NotComplete"],
                   data[snap][f]["Diffuse_Population_NotComplete"],
                   data[snap][f]["okinds"],
                   data[snap][f]["Weight"],
                   f, snap, orientation, "Total", "default",
                   data[snap][f]["Complete_Luminosity"],
                   data[snap][f]["Complete_Mass"], weight_norm)
        print("--------------------------- HDR", snap,
              "---------------------------")
        hdr_comp(data[snap][f]["HDR"], data[snap][f]["HLR_0.5"],
                 intr_data[snap][f]["HLR_0.5"], data[snap][f]["Weight"],
                 data[snap][f]["Compact_Population_Complete"],
                 data[snap][f]["Diffuse_Population_Complete"],
                 data[snap][f]["Compact_Population_NotComplete"],
                 data[snap][f]["Diffuse_Population_NotComplete"],
                 f, orientation, snap, "Intrinsic", "default", weight_norm)
        print(
            "--------------------------- Intrinsic", snap,
            "---------------------------")
        size_lumin_intrinsic(intr_data[snap][f]["HLR_Pixel_0.5"],
                             intr_data[snap][f]["Image_Luminosity"],
                             data[snap][f]["Weight"],
                             intr_data[snap][f]["Compact_Population_Complete"],
                             intr_data[snap][f]["Diffuse_Population_Complete"],
                             intr_data[snap][f][
                                 "Compact_Population_NotComplete"],
                             intr_data[snap][f][
                                 "Diffuse_Population_NotComplete"],
                             f, snap, "pix", orientation, "Intrinsic",
                             "default", weight_norm)
        size_lumin_intrinsic(intr_data[snap][f]["HLR_0.5"],
                             intr_data[snap][f]["Luminosity"],
                             data[snap][f]["Weight"],
                             intr_data[snap][f]["Compact_Population_Complete"],
                             intr_data[snap][f]["Diffuse_Population_Complete"],
                             intr_data[snap][f][
                                 "Compact_Population_NotComplete"],
                             intr_data[snap][f][
                                 "Diffuse_Population_NotComplete"],
                             f, snap, "part", orientation, "Intrinsic",
                             "default", weight_norm)
        print("--------------------------- Comp", snap,
              "---------------------------")
        size_comp(f, snap, intr_data[snap][f]["HLR_0.5"],
                  intr_data[snap][f]["HLR_Pixel_0.5"], data[snap][f]["Weight"],
                  intr_data[snap][f]["Compact_Population_Complete"],
                  intr_data[snap][f]["Diffuse_Population_Complete"],
                  intr_data[snap][f]["Compact_Population_NotComplete"],
                  intr_data[snap][f]["Diffuse_Population_NotComplete"],
                  weight_norm, orientation, "Intrinsic", "default")
        size_comp(f, snap, data[snap][f]["HLR_0.5"],
                  data[snap][f]["HLR_Pixel_0.5"], data[snap][f]["Weight"],
                  data[snap][f]["Compact_Population_Complete"],
                  data[snap][f]["Diffuse_Population_Complete"],
                  data[snap][f]["Compact_Population_NotComplete"],
                  data[snap][f]["Diffuse_Population_NotComplete"],
                  weight_norm, orientation, "Total", "default")

        size_comp_smooth(f, snap, data[snap][f]["HLR_Pixel_0.5"],
                         data[snap][f]["HLR_Pixel_0.5_No_Smooth"],
                         data[snap][f]["Weight"],
                         data[snap][f]["Compact_Population_Complete"],
                         data[snap][f]["Diffuse_Population_Complete"],
                         data[snap][f]["Compact_Population_NotComplete"],
                         data[snap][f]["Diffuse_Population_NotComplete"],
                         weight_norm, orientation, "Total", "default")
        size_comp_smooth(f, snap, intr_data[snap][f]["HLR_Pixel_0.5"],
                         intr_data[snap][f]["HLR_Pixel_0.5_No_Smooth"],
                         data[snap][f]["Weight"],
                         intr_data[snap][f]["Compact_Population_Complete"],
                         intr_data[snap][f]["Diffuse_Population_Complete"],
                         intr_data[snap][f]["Compact_Population_NotComplete"],
                         intr_data[snap][f]["Diffuse_Population_NotComplete"],
                         weight_norm, orientation, "Intrinsic", "default")

        size_comp_smooth_part(f, snap, data[snap][f]["HLR_0.5"],
                              data[snap][f]["HLR_Pixel_0.5_No_Smooth"],
                              data[snap][f]["Weight"],
                              data[snap][f]["Compact_Population_Complete"],
                              data[snap][f]["Diffuse_Population_Complete"],
                              data[snap][f]["Compact_Population_NotComplete"],
                              data[snap][f]["Diffuse_Population_NotComplete"],
                              weight_norm, orientation, "Total", "default")
        size_comp_smooth_part(f, snap, intr_data[snap][f]["HLR_0.5"],
                              intr_data[snap][f]["HLR_Pixel_0.5_No_Smooth"],
                              data[snap][f]["Weight"],
                              intr_data[snap][f][
                                  "Compact_Population_Complete"],
                              intr_data[snap][f][
                                  "Diffuse_Population_Complete"],
                              intr_data[snap][f][
                                  "Compact_Population_NotComplete"],
                              intr_data[snap][f][
                                  "Diffuse_Population_NotComplete"],
                              weight_norm, orientation, "Intrinsic", "default")

img_size_comp(filters[0], regions, snaps[-1], weight_norm,
              orientation, "Total", "default")
