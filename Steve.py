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
        starZ.extend(hdf[reg][snap]["Galaxy"]["Metallicity"]["MassWeightedStellarZ"][...])
        gasZ.extend(hdf[reg][snap]["Galaxy"]["Metallicity"]["MassWeightedGasZ"][...])

sfr_10 = np.array(sfr_10)
sfr_inst = np.array(sfr_inst)
cent_sat = np.array(cent_sat)
smass = np.array(smass)
starZ = np.array(starZ)
gasZ = np.array(gasZ)
cent_sat[cent_sat > 0] = 1

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for ax in [ax1, ax2, ax3, ax4]:
    ax.loglog()
    ax.set_ylim(1, 10 ** 4)
    ax.set_xlim(1, 10 ** 4)

sinds = np.argsort(smass)
ax1.scatter(sfr_10[sinds], sfr_inst[sinds], c=smass[sinds], marker="o", s=4, cmap=cmr.apple,
            norm=LogNorm())
sinds = np.argsort(cent_sat)
ax2.scatter(sfr_10[sinds], sfr_inst[sinds], c=cent_sat[sinds], marker="o", s=4, cmap="bwr",
            vmin=0, vmax=1)
sinds = np.argsort(starZ)
ax3.scatter(sfr_10[sinds], sfr_inst[sinds], c=starZ[sinds], marker="o", s=4, cmap=cmr.amethyst)
sinds = np.argsort(gasZ)
ax4.scatter(sfr_10[sinds], sfr_inst[sinds], c=gasZ[sinds], marker="o", s=4, cmap=cmr.cosmic)

fig.savefig("plots/sfr_comp.png", bbox_inches="tight")



