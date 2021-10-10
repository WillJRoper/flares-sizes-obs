import h5py
import numpy as np
import pandas as pd
from plt_half_dust_radius_comp import hdr_comp
from plt_mass_lumin import mass_lumin
from plt_size_evo_violin import size_evo_violin
from plt_size_lumin_fitgrid import fit_size_lumin_grid
from plt_size_lumin_grid import size_lumin_grid
from plt_size_lumin_intrinsic import size_lumin_intrinsic

# Set orientation
orientation = "sim"

snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
         '010_z005p000']
all_snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
             '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '011_z004p770']

# Define filter
# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
#                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
filters = ['FAKE.TH.' + f for f in ['FUV', 'MUV', 'NUV']]

keys = ["Mass", "Image_Luminosity", "HLR_0.5",
        "HLR_Pixel_0.5", "Luminosity", "HDR", "Images", "nStar"]

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

for reg, snap in reg_snaps:

    try:
        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Total",
                                                        orientation),
            "r")
    except OSError as e:
        print(e)
        continue

    data.setdefault(snap, {})
    intr_data.setdefault(snap, {})

    for f in filters:

        print(reg, snap, f)

        data[snap].setdefault(f, {})
        intr_data[snap].setdefault(f, {})

        for key in keys:
            data[snap][f].setdefault(key, []).extend(hdf[f][key][...])

        data[snap][f].setdefault("Weight", []).extend(
            np.full(hdf[f]["Mass"][...].size, weights[int(reg)]))

    hdf.close()

    try:
        hdf = h5py.File(
            "data/flares_sizes_{}_{}_{}_{}.hdf5".format(reg, snap, "Intrinsic",
                                                        orientation),
            "r")
    except OSError as e:
        print(e)
        continue

    for f in filters:

        surf_dens = hdf[f]["Image_Luminosity"][...] \
                    / (np.pi * (2 * hdf[f]["HLR_0.5"][...]) ** 2)

        intr_data[snap][f].setdefault("Inner_Surface_Density",
                                      []).extend(surf_dens)

        for key in keys:
            intr_data[snap][f].setdefault(key, []).extend(hdf[f][key][...])

    hdf.close()

for snap in all_snaps:

    for f in filters:

        print(snap, f)

        for key in data[snap][f].keys():
            data[snap][f][key] = np.array(data[snap][f][key])

        for key in intr_data[snap][f].keys():
            intr_data[snap][f][key] = np.array(intr_data[snap][f][key])

        okinds = np.logical_and(
            intr_data[snap][f]["Inner_Surface_Density"] > 10 ** 26,
            intr_data[snap][f]["nStar"] > 100)

        data[snap][f]["okinds"] = okinds
        intr_data[snap][f]["okinds"] = okinds

        compact_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) >= 10 ** 29
        diffuse_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) < 10 ** 29

        data[snap][f]["Compact_Population"] = compact_pop
        data[snap][f]["Diffuse_Population"] = diffuse_pop
        intr_data[snap][f]["Compact_Population"] = compact_pop
        intr_data[snap][f]["Diffuse_Population"] = diffuse_pop

size_lumin_grid(data, snaps, filters, orientation, "Total", "default", "pix")
fit_size_lumin_grid(data, snaps, filters, orientation, "Total", "default",
                    "pix")

for f in filters:
    size_evo_violin(data, intr_data, snaps, f, "pix", "sim")
    for snap in snaps:
        mass_lumin(intr_data[snap][f]["Mass"],
                   intr_data[snap][f]["Luminosity"],
                   intr_data[snap][f]["okinds"],
                   intr_data[snap][f]["Diffuse_Population"],
                   intr_data[snap][f]["Compact_Population"],
                   data[snap][f]["Weight"],
                   f, snap, orientation, "Intrinsic", "default")
        hdr_comp(data[snap][f]["HDR"], data[snap][f]["HLR_0.5"],
                 intr_data[snap][f]["HLR_0.5"], data[snap][f]["Weight"],
                 data[snap][f]["okinds"], data[snap][f]["Compact_Population"],
                 data[snap][f]["Diffuse_Population"], f,
                 orientation, snap, "Intrinsic", "default")
        size_lumin_intrinsic(intr_data[snap][f]["HLR_Pixel_0.5"],
                             intr_data[snap][f]["Image_Luminosity"],
                             intr_data[snap][f]["Weight"],
                             intr_data[snap][f]["okinds"],
                             intr_data[snap][f]["Compact_Population"],
                             intr_data[snap][f]["Diffuse_Population"], f, snap,
                             "pix", orientation, "default")
