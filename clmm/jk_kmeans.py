from kmeans_radec import kmeans_sample, find_nearest
from clmm.dataops import make_stacked_radial_profile
import numpy as np

def radec_to_xyz(coords):
    ra_rad = np.radians(coords[:, 0])
    dec_rad = np.radians(coords[:, 1])
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])


def jk_regions(X, njk):
    km = kmeans_sample(X, ncen=njk)
    labels = find_nearest(X, km.centers)
    return labels, km.centers


def jk_cov(clusterensemble, njk):
    X = np.vstack((clusterensemble['ra'], clusterensemble['dec'])).T
    labels, centers = jk_regions(X, njk=njk)
    unique_labels = np.unique(labels)
    gt_jack, gx_jack = [], []
    for drop_region in unique_labels:
        mask = np.isin(labels, drop_region, invert=True)
        gt, gx = make_stacked_radial_profile(
            clusterensemble['radius'][mask],
            clusterensemble['W_l'][mask],
            [clusterensemble['g_t'][mask], clusterensemble['g_x'][mask]])[1]
        gt_jack.append(gt)
        gx_jack.append(gx)
    n_jack = unique_labels.size
    coeff = (n_jack - 1)**2 / n_jack
    tan_jk = coeff * np.cov(np.transpose(gt_jack), bias=False, ddof=0)
    cross_jk = coeff * np.cov(np.transpose(gx_jack), bias=False, ddof=0)
    cart_centers = radec_to_xyz(centers) 
    return tan_jk, cross_jk, cart_centers


