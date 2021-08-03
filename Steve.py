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
import h5py


sns.set_context("paper")
sns.set_style('whitegrid')


hdf = h5py.File("/cosma7/data/dp004/dc-payy1/my_files/"
                "flares_pipeline/data/flares.hdf5", "r")

for reg in hdf.keys():
    for snap in hdf[reg].keys():
        print(reg, snap, hdf[snap]["Galaxy"].keys())


