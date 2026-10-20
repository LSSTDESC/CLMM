"""Functions to define sky regions for jackknife resampling"""

import numpy as np


def _radec_to_xyz(ra, dec):
    """Convert sky coordinates to cartesian coordinates on the unit sphere.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination in degrees.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates, shape (n, 3).
    """
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    return np.column_stack(
        [np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)]
    )


def compute_kmeans_centers(ra, dec, n_regions, seed=None, maxiter=100):
    """Define `n_regions` jackknife regions on the sky with k-means clustering of the points
    (`ra`, `dec`) on the sphere. Requires the `kmeans_radec` package
    (https://github.com/esheldon/kmeans_radec).

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination of the points, in degrees.
    n_regions : int
        Number of regions, at most the number of points.
    seed : int, None, optional
        Seed for the random first-pass sample and initial centers. A local random generator
        is used, so global random states are not changed. Default: None
    maxiter : int, optional
        Maximum number of k-means iterations. Default: 100

    Returns
    -------
    numpy.ndarray
        Region centers (ra, dec) in degrees, shape (`n_regions`, 2).
    """
    try:
        from kmeans_radec import KMeans  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "compute_kmeans_centers requires kmeans_radec: "
            "pip install git+https://github.com/esheldon/kmeans_radec.git"
        ) from err
    coords = np.column_stack([ra, dec])
    if not 0 < n_regions <= len(coords):
        raise ValueError(
            f"n_regions ({n_regions}) must be between 1 and the number of points ({len(coords)})"
        )
    # Two passes as in kmeans_radec.kmeans_sample, which draws its samples from the global
    # `random` state: a first pass on a random subsample, then a pass on all the points.
    # kmeans_radec's first-pass sample size, max(2 sqrt(N), 10 n_regions),
    # can exceed the number of points, so cap it
    rng = np.random.default_rng(seed)
    nsample = int(min(len(coords), max(2 * np.sqrt(len(coords)), 10 * n_regions)))
    sample = coords[rng.choice(len(coords), nsample, replace=False)]
    start = coords[rng.choice(len(coords), n_regions, replace=False)]
    kmeans = KMeans(start, verbose=0)
    kmeans.run(sample, maxiter=maxiter)
    kmeans = KMeans(kmeans.get_centers(), verbose=0)
    kmeans.run(coords, maxiter=maxiter)
    return kmeans.get_centers()


def assign_kmeans_regions(ra, dec, centers):
    """Assign each point (`ra`, `dec`) to the region of its nearest center on the sphere.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination of the points, in degrees.
    centers : array_like
        Region centers (ra, dec) in degrees, shape (n_regions, 2), e.g. from
        `compute_kmeans_centers`.

    Returns
    -------
    numpy.ndarray
        Index of the region of each point.
    """
    centers = np.atleast_2d(centers)
    # the nearest center has the largest scalar product on the unit sphere
    return np.argmax(_radec_to_xyz(ra, dec) @ _radec_to_xyz(*centers.T).T, axis=1)
