import h5py
import numpy as np
import pandas as pd
from plt_size_lumin_grid import size_lumin_grid

# Set orientation
orientation = "sim"

snaps = ['006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
         '010_z005p000']

# Define filter
# filters = ['FAKE.TH.' + f
#            for f in ['FUV', 'MUV', 'NUV', 'U', 'B',
#                      'V', 'R', 'I', 'Z', 'Y', 'J', 'H']]
filters = ['FAKE.TH.' + f for f in ['FUV', 'MUV', 'NUV']]

keys = ["Mass", "Image_Luminosity", "HLR_0.5",
        "HLR_Pixel_0.5", "Luminosity", "HDR", "Images"]

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

    for snap in snaps:
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

for snap in snaps:

    for f in filters:

        okinds = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) > 10 ** 26

        for key in data[snap][f].keys():
            data[snap][f][key] = np.array(data[snap][f][key])[okinds]

        for key in intr_data[snap][f].keys():
            intr_data[snap][f][key] = np.array(intr_data[snap][f][key])[okinds]

        compact_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) >= 10 ** 29
        diffuse_pop = np.array(
            intr_data[snap][f]["Inner_Surface_Density"]) < 10 ** 29

        data[snap][f]["Compact_Population"] = compact_pop
        data[snap][f]["Diffuse_Population"] = diffuse_pop
        intr_data[snap][f]["Compact_Population"] = compact_pop
        intr_data[snap][f]["Diffuse_Population"] = diffuse_pop

size_lumin_grid(data, snaps, filters, orientation, "Total", "default", "pix")
