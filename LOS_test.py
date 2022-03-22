import numpy as np
from scipy.spatial import cKDTree, distance
import astropy.units as u


kinp = np.load('kernel_sph-anarchy.npz',
               allow_pickle=True)
lkernel = kinp['kernel']
header = kinp['header']
kbins = header.item()['bins']


def get_Z_LOS_orig(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                   dimens=(0, 1, 2)):
    """

    Compute the los metal surface density (in g/cm^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """

    conv = (u.solMass / u.Mpc ** 2).to(u.solMass / u.pc ** 2)

    n = s_cood.shape[0]
    Z_los_SD = np.zeros(n)

    # Fixing the observer direction as z-axis. Use make_faceon()
    # for changing the
    # particle orientation to face-on
    xdir, ydir, zdir = dimens
    for ii in range(n):
        thisspos = s_cood[ii]
        ok = np.where(g_cood[:, zdir] > thisspos[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]

        # Get radii and divide by smooting length
        b = np.linalg.norm(thisgpos[:, (xdir, ydir)]
                           - thisspos[((xdir, ydir), )],
                           axis=-1)
        boverh = b / thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok] * thisgZ[ok] / (
                thisgsml[ok] * thisgsml[
            ok])) * kernel_vals) * conv  # in units of Msun/pc^2

    return Z_los_SD


def get_Z_LOS_kd(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins,
                 dimens=(0, 1, 2)):
    """

    Compute the los metal surface density (in g/cm^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length
        dimens (tuple: int): tuple of xyz coordinates

    """

    # Generalise dimensions (function assume LOS along z-axis)
    xdir, ydir, zdir = dimens

    # Get how many stars
    nstar = s_cood.shape[0]

    # Conversion factor
    conv = (u.solMass / u.Mpc ** 2).to(u.solMass / u.pc ** 2)

    # Lets build the kd tree from star positions
    tree = cKDTree(s_cood[:, (xdir, ydir)])

    # Query the tree for all gas particles (can now supply multiple rs!)
    query = tree.query_ball_point(g_cood[:, (xdir, ydir)], r=g_sml, p=1)

    # Now we just need to collect each stars neighbours
    star_gas_nbours = {s: [] for s in range(nstar)}
    for g_ind, sparts in enumerate(query):
        for s_ind in sparts:
            star_gas_nbours[s_ind].append(g_ind)

    # Initialise line of sight metal density
    Z_los_SD = np.zeros(nstar)

    # Loop over stars
    for s_ind in range(nstar):

        # Extract gas particles to consider
        g_inds = star_gas_nbours.pop(s_ind)

        # Extract data for these particles
        thisspos = s_cood[s_ind]
        thisgpos = g_cood[g_inds]
        thisgsml = g_sml[g_inds]
        thisgZ = g_Z[g_inds]
        thisgmass = g_mass[g_inds]

        # We only want to consider particles "in-front" of the star
        ok = np.where(thisgpos[:, zdir] > thisspos[zdir])[0]
        thisgpos = thisgpos[ok]
        thisgsml = thisgsml[ok]
        thisgZ = thisgZ[ok]
        thisgmass = thisgmass[ok]

        # Get radii and divide by smooting length
        b = np.linalg.norm(thisgpos[:, (xdir, ydir)]
                           - thisspos[((xdir, ydir), )],
                           axis=-1)
        boverh = b / thisgsml

        # Apply kernel
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh])

        # Finally get LOS metal surface density in units of Msun/pc^2
        Z_los_SD[s_ind] = np.sum((thisgmass * thisgZ
                                  / (thisgsml * thisgsml))
                                 * kernel_vals) * conv

    return Z_los_SD


# import sys
# sys.path.insert(0,'/Users/willroper/Documents/University/FLARES/flares-sizes-obs')
# import os
# os.chdir("flares-sizes-obs")
# from LOS_test import *
# kinp = np.load('kernel_sph-anarchy.npz', allow_pickle=True)
# lkernel = kinp['kernel']
# header = kinp['header']
# kbins = header.item()['bins']
# s_cood = np.random.normal(0, 5, (10000, 3))
# g_cood = np.random.normal(0, 7, (20000, 3))
# g_sml = np.ones(20000) * np.random.uniform(0.2, 1.2, (20000))
# g_mass = np.ones(20000) * np.random.uniform(0.5, 1.2, (20000))
# g_Z = np.ones(20000) * np.random.uniform(0.5, 1.2, (20000))
# %timeit get_Z_LOS_orig(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)
# %timeit get_Z_LOS_kd(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)
# res_orig = get_Z_LOS_orig(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)
# res_kd = get_Z_LOS_kd(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins)
# print(np.sum(res_orig), np.sum(res_kd), np.sum(res_orig) / np.sum(res_kd) * 100)
