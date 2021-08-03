#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import cmasher as cmr

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import seaborn as sns
import h5py


sns.set_context("paper")
sns.set_style('whitegrid')


hdf = h5py.File("/cosma7/data/dp004/dc-payy1/my_files/"
                "flares_pipeline/data/flares.hdf5", "r")

sfr_inst = []
sfr_10 = []
cent_sat = []
smass = []
starZ = []
gasZ = []

for reg in hdf.keys():
    for snap in hdf[reg].keys():
        print(reg, snap)

        sfr_10.extend(hdf[reg][snap]["Galaxy"]["SFR"]["SFR_10"][...])
        sfr_inst.extend(hdf[reg][snap]["Galaxy"]["SFR_inst_30"][...])
        cent_sat.extend(hdf[reg][snap]["Galaxy"]["SubGroupNumber"][...])
        smass.extend(hdf[reg][snap]["Galaxy"]["Mstar_30"][...] * 10**10)
        starZ.extend(hdf[reg][snap]["Galaxy"]["Metallicity"]["MassWeightedStellarZ "][...])
        gasZ.extend(hdf[reg][snap]["Galaxy"]["Metallicity"]["MassWeightedGasZ"][...])

cent_sat = np.array(cent_sat)
cent_sat[cent_sat > 0] = 1

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for ax in [ax1, ax2, ax3, ax4]:
    ax.loglog()

ax1.scatter(sfr_10, sfr_inst, c=smass, marker="o", cmap=cmr.apple,
            norm=LogNorm)
ax2.scatter(sfr_10, sfr_inst, c=cent_sat, marker="o", cmap="bwr",
            vmin=0, vmax=1)
ax3.scatter(sfr_10, sfr_inst, c=starZ, marker="o", cmap=cmr.amethyst)
ax4.scatter(sfr_10, sfr_inst, c=gasZ, marker="o", cmap=cmr.cosmic)

fig.savefig("plots/sfr_comp.png", bbox_inches="tight")



