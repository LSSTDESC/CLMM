import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from kmeans_radec.test import plot_centers
from kmeans_radec import KMeans, kmeans_sample, find_nearest

PLOTS_DIR = "./plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# randomly draw cluster positions over the full sky
N = 30
ra = np.random.random(N) * 360  # from 0 to 360 deg
sindec = np.random.random(N) * 2 - 1
dec = np.arcsin(sindec) * 180 / np.pi  # from -90 to 90 deg
jk_n_side = 16
# CLMM's jackknife regions are HEALPix pixels (not kmeans patches like TreeCorr),
# and are not stored on the object, so recompute them here to visualize the full
# extent of each region using healpy's own map plotting (cartview).
jk_pixels = hp.ang2pix(
    jk_n_side, ra, dec, nest=True, lonlat=True
)
jk_unique_pixels = np.unique(jk_pixels)
jk_center_ra, jk_center_dec = hp.pix2ang(jk_n_side, jk_unique_pixels, nest=True, lonlat=True)
jk_map = np.full(hp.nside2npix(jk_n_side), hp.UNSEEN)
jk_map[jk_unique_pixels] = np.arange(jk_unique_pixels.size)

fig = plt.figure(figsize=(12, 5))

# Left panel: healpy manages its own projection axes internally, so instead of
# handing it a matplotlib Axes, place it into a subplot slot of our figure via
# fig= and sub= (same (nrows, ncols, index) convention as plt.subplot).
hp.cartview(
    jk_map,
    fig=fig.number,
    sub=(1, 2, 1),
    nest=True,
    cmap="tab20",
    cbar=False,
    title="Jackknife patches (HEALPix pixels)",
)
hp.graticule()
hp.projscatter(ra, dec, lonlat=True, c="k", s=10)

njk = 3 
X = np.vstack((ra, dec)).T
km = kmeans_sample(X, ncen=njk, method='slow')

ax2= fig.add_subplot(1, 2, 2)
plot_centers(km.centers, ax2, y_min=-90, y_max=90)
ax2.scatter(ra, dec, c="r", s=30)
ax2.set_title("kmeans_radec Voronoi cells")
ax2.set_axis_off()

fig.savefig(os.path.join(PLOTS_DIR, "hp_vs_kmeans.png"))
plt.show()

print('----- Distances ------')
print(km.distances)
labels = find_nearest(X, km.centers)
print('--- labels ---')
print(labels)
