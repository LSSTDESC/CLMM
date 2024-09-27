"""Functions to generate mock source galaxy distributions to demo lensing code"""

import warnings
import numpy as np

from scipy.special import erfc

from astropy import units as u
from astropy.coordinates import SkyCoord

from ..gcdata import GCData
from ..theory import compute_tangential_shear, compute_convergence
from ..utils import (
    convert_units,
    compute_lensed_ellipticity,
    validate_argument,
    _draw_random_points_from_distribution,
    gaussian,
    _validate_coordinate_system,
)
from ..redshift import distributions as zdist


def generate_galaxy_catalog(
    cluster_m,
    cluster_z,
    cluster_c,
    cosmo,
    zsrc,
    cluster_ra=0.0,
    cluster_dec=0.0,
    delta_so=200,
    massdef="mean",
    halo_profile_model="nfw",
    zsrc_min=None,
    zsrc_max=7.0,
    field_size=8.0,
    shapenoise=None,
    mean_e_err=None,
    photoz_sigma_unscaled=None,
    nretry=5,
    ngals=None,
    ngal_density=None,
    pz_bins=101,
    pz_quantiles_conf=(5, 31),
    pzpdf_type="shared_bins",
    coordinate_system="euclidean",
    validate_input=True,
):
    r"""Generates a mock dataset of sheared background galaxies.

    We build galaxy catalogs following a series of steps.

    1. Draw true redshifts of the source galaxy population. This step is described by the
    parameters `zsrc` and `zsrc_max`. `zsrc` can be a `float` in which case every source is
    at the given redshift or a `str` describing a specific model to use for the source
    distribution. Currently, the only supported model for source galaxy distribution is that
    of Chang et al. 2013 arXiv:1305.0793. When a model is used to describe the distribution,
    `zsrc_max` is the maximum allowed redshift of a source galaxy.

    2. Apply photometric redshift errors to the source galaxy population. This step is
    described by the parameter `photoz_sigma_unscaled`. If this parameter is set to a float,
    we add Gaussian uncertainty to the source redshift

    .. math::
        z \sim \mathcal{N}\left(z^{\rm true},
        \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true}) \right)

    We additionally include two columns in the output catalog, `pzbins` and `pzpdf` which
    desribe the photo-z distribution as a Gaussian centered at :math:`z^{\rm true}` with a
    width :math:`\sigma_{\rm photo-z} = \sigma_{\rm photo-z}^{\rm unscaled}(1+z^{\rm true})`

    If `photoz_sigma_unscaled` is `None`, the `z` column in the output catalog is the true
    redshift.

    3. Draw galaxy positions. Positions are drawn in a square box around the lens position (with a
    default side length of 8 Mpc) at the lens redhsift. We then convert to right ascension and
    declination using the cosmology defined in `cosmo`.

    4. We predict the reduced tangential shear of each using the radial distances of each source
    from the lens, the source redshifts, and the lens mass, concentration, and redshift. In the
    given cosmology for an NFW halo. The reduced tangential shear is then transformed into `g1`
    and `g2` components.

    5. If the `shapenoise=True`, intrinsic ellipticities (1,2) components are drawn from a
    Gaussian distribution of width of `shapenoise`. These ellipticities components are then
    combined with `g1` and `g2` to provide lensed ellipticies `e1` and `e2`. If
    `shapenoise=False`, `g1` and `g2` are directly used as ellipticity components.


    If the shape noise parameter is high, we may draw nonsensical values for ellipticities. We
    ensure that we does not return any nonsensical values for derived properties. We re-draw
    all galaxies with e1 or e2 outside the bounds of [-1, 1]. After 5 (default) attempts to
    re-draw these properties, we return the catalog as is and throw a warning.

    Parameters
    ----------
    cluster_m : float
        Cluster mass in Msun
    cluster_z : float
        Cluster redshift
    cluster_c : float
        Cluster concentration in the same mass definition as delta_so
    cosmo: clmm.Cosmology, optional
        Cosmology object.
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.

        * `float` : All sources galaxies at this fixed redshift;
        * `str` : Draws individual source gal redshifts from predefined distribution. Options are:

            * `chang13` - Chang et al. 2013 (arXiv:1305.0793);
            * `desc_srd` - LSST/DESC Science Requirement Document (arxiv:1809.01669);

    delta_so : float, optional
        Overdensity density contrast used to compute the cluster mass and concentration. The
        spherical overdensity mass is computed as the mass enclosed within the radius
        :math:`R_{\Delta{\rm SO}}` where the mean matter density is :math:`\Delta_{\rm SO}` times
        the mean (or critical, depending on the massdef keyword) density of the Universe at the
        cluster redshift
        :math:`M_{\Delta{\rm SO}}=4/3\pi\Delta_{\rm SO}\rho_{m}(z_{\rm lens})R_{\Delta{\rm SO}}^3`
    massdef : string, optional
        Definition the mass overdensity with respect to the 'mean' or 'critical' density of the
        universe. Default is 'mean' as it works for all theory backends. The NumCosmo and CCL
        backends also allow the use of 'critical'.  (letter case independent)
    halo_profile_model : string, optional
        Halo density profile. Default is 'nfw', which works for all theory backends. The NumCosmo
        backend allow for more options, e.g. 'einasto' or 'burkert' profiles (letter case
        independent).
    zsrc_min : float, optional
        The minimum true redshift of the sources. If photoz errors are included, the observed
        redshift may be smaller than zsrc_min.
    zsrc_max : float, optional
        The maximum true redshift of the sources, apllied when galaxy redshifts are drawn from a
        redshift distribution. If photoz errors are included, the observed redshift may be larger
        than zsrc_max.
    field_size : float, optional
        The size of the field (field_size x field_size) to be simulated.
        Proper distance in Mpc  at the cluster redshift.
    shapenoise : float, optional
        If set, applies Gaussian shape noise to the galaxy shapes with a width set by `shapenoise`
    mean_e_err : float, optional
        Mean per-component ellipticity uncertainty. Currently,
        individual uncertainties are drawn from a uniform distribution
        in the range [0.9,1.1]*mean_e_err. If not provided, the output
        table will not include this column.
    photoz_sigma_unscaled : float, optional
        If set, applies photo-z errors to source redshifts
    pz_bins: int, array
        Photo-z pdf bins in the given range. If int, the limits are set automatically.
        If is array, must be the bin edges.
    pz_quantiles_conf: tuple
        Configuration for quantiles when `pzpdf_type='quantiles'`. Must be with the format
        `(max_sigma_dev, num_points)`, which is used as
        `sigma_steps = np.linspace(-max_sigma_dev, max_sigma_dev, num_points)`
    pzpdf_type: str, None
        Type of photo-z pdf to be stored, options are:
            `None` - does not store PDFs;
            `'shared_bins'` - single binning for all galaxies
            `'individual_bins'` - individual binning for each galaxy
            `'quantiles'` - quantiles of PDF
    nretry : int, optional
        The number of times that we re-draw each galaxy with non-sensical derived properties
    ngals : float, optional
        Number of galaxies to generate
    ngal_density : float, optional
        The number density of galaxies (in galaxies per square arcminute, from z=0 to z=infty).
        The number of galaxies to be drawn will then depend on the redshift distribution and
        user-defined redshift range.  If specified, the ngals argument will be ignored.
    coordinate_system : str, optional
        Coordinate system of the ellipticity components. Must be either 'celestial' or
        euclidean'. See https://doi.org/10.48550/arXiv.1407.7676 section 5.1 for more details.
        Default is 'euclidean'.

    validate_input: bool
        Validade each input argument

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of source galaxies with drawn and derived properties required for lensing studies

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    # pylint: disable=too-many-arguments
    # Too many local variables (25/15)
    # pylint: disable=R0914

    if validate_input:
        validate_argument(locals(), "cluster_m", float, argmin=0, eqmin=True)
        validate_argument(locals(), "cluster_z", float, argmin=0, eqmin=True)
        validate_argument(locals(), "cluster_c", float, argmin=0, eqmin=True)
        validate_argument(
            locals(),
            "cluster_dec",
            float,
            argmin=-90,
            argmax=90,
            eqmin=True,
            eqmax=True,
        )
        validate_argument(
            locals(),
            "cluster_ra",
            float,
            argmin=-360.0,
            argmax=360.0,
            eqmin=True,
            eqmax=True,
        )
        validate_argument(locals(), "zsrc", (float, str))
        validate_argument(locals(), "delta_so", float, argmin=0, eqmin=True)
        validate_argument(locals(), "massdef", str)
        validate_argument(locals(), "halo_profile_model", str)
        validate_argument(locals(), "zsrc_min", float, argmin=0, eqmin=True, none_ok=True)
        validate_argument(locals(), "zsrc_max", float, argmin=0, eqmin=True)
        validate_argument(locals(), "field_size", float, argmin=0, eqmin=True)
        validate_argument(locals(), "shapenoise", float, argmin=0, none_ok=True)
        validate_argument(locals(), "mean_e_err", float, argmin=0, none_ok=True)
        validate_argument(locals(), "photoz_sigma_unscaled", float, argmin=0, none_ok=True)
        validate_argument(locals(), "nretry", int)
        validate_argument(locals(), "ngals", float, none_ok=True)
        validate_argument(locals(), "ngal_density", float, none_ok=True)
        validate_argument(locals(), "pz_bins", (int, "array"))
        _validate_coordinate_system(locals(), "coordinate_system", str)

    if zsrc_min is None:
        zsrc_min = cluster_z + 0.1

    params = {
        "cluster_m": cluster_m,
        "cluster_z": cluster_z,
        "cluster_c": cluster_c,
        "cluster_ra": cluster_ra,
        "cluster_dec": cluster_dec,
        "cosmo": cosmo,
        "delta_so": delta_so,
        "zsrc": zsrc,
        "massdef": massdef,
        "halo_profile_model": halo_profile_model,
        "zsrc_min": zsrc_min,
        "zsrc_max": zsrc_max,
        "shapenoise": shapenoise,
        "mean_e_err": mean_e_err,
        "photoz_sigma_unscaled": photoz_sigma_unscaled,
        "pz_bins": pz_bins,
        "pz_quantiles_conf": pz_quantiles_conf,
        "field_size": field_size,
        "pzpdf_type": pzpdf_type,
        "coordinate_system": coordinate_system,
    }

    if ngals is None and ngal_density is None:
        err = (
            "Either the number of galaxies 'ngals' or the galaxy density"
            " 'ngal_density' keyword must be specified"
        )
        raise ValueError(err)

    if ngals is not None and ngal_density is not None:
        err = "The 'ngals' and 'ngal_density' keywords cannot both be set. Please use only one."
        raise ValueError(err)

    if ngal_density is not None:
        # Compute the number of galaxies to be drawn
        ngals = _compute_ngals(
            ngal_density,
            field_size,
            cosmo,
            cluster_z,
            zsrc,
            zsrc_min=zsrc_min,
            zsrc_max=zsrc_max,
        )

    galaxy_catalog = _generate_galaxy_catalog(ngals=ngals, **params)
    # Check for bad galaxies and replace them
    nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
    ntry = 0
    # Prep bins for replacement
    if photoz_sigma_unscaled is not None and pzpdf_type == "shared_bins" and nbad > 0:
        params["pz_bins"] = galaxy_catalog.pzpdf_info["zbins"]
    while (nbad > 0) and (ntry < nretry):
        replacements = _generate_galaxy_catalog(ngals=nbad, **params)
        # galaxy_catalog[badids] = replacements
        for badid, replacement in zip(badids, replacements):
            for col in galaxy_catalog.colnames:
                galaxy_catalog[col][badid] = replacement[col]
        nbad, badids = _find_aphysical_galaxies(galaxy_catalog, zsrc_min)
        ntry += 1

    # Final check to see if there are bad galaxies left
    if nbad > 1:
        warnings.warn(f"Not able to remove {nbad} aphysical objects after {nretry} iterations")

    # Now that the catalog is final, add an id column
    galaxy_catalog["id"] = np.arange(ngals)
    return galaxy_catalog


def _compute_ngals(ngal_density, field_size, cosmo, cluster_z, zsrc, zsrc_min=None, zsrc_max=None):
    """
    A private function that computes the number of galaxies to draw given the user-defined
    field size, galaxy density, cosmology, cluster redshift, galaxy redshift distribution
    and requested redshift range.

    Parameters
    ----------
    ngal_density : float, optional
        The number density of galaxies (in galaxies per square arcminute, from z=0 to z=infty).
        The number of galaxies to be drawn will then depend on the redshift distribution and
        user-defined redshift range.  If specified, the ngals argument will be ignored.
    field_size : float, optional
        The size of the field (field_size x field_size) to be simulated.
        Proper distance in Mpc  at the cluster redshift.
    cosmo: clmm.Cosmology, optional
        Cosmology object.
    cluster_z : float
        Cluster redshift
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.

        * `float` : All sources galaxies at this fixed redshift;
        * `str` : Draws individual source gal redshifts from predefined distribution. Options are:

            * `chang13` - Chang et al. 2013 (arXiv:1305.0793);
            * `desc_srd` - LSST/DESC Science Requirement Document (arxiv:1809.01669);

    zsrc_min : float, optional
        The minimum true redshift of the sources. If photoz errors are included, the observed
        redshift may be smaller than zsrc_min.
    zsrc_max : float, optional
        The maximum true redshift of the sources, apllied when galaxy redshifts are drawn from a
        redshift distribution. If photoz errors are included, the observed redshift may be larger
        than zsrc_max.

    Returns
    -------
    ngals : int
        Number of galaxies to be generated.
    """
    field_size_arcmin = convert_units(field_size, "Mpc", "arcmin", redshift=cluster_z, cosmo=cosmo)
    ngals = ngal_density * field_size_arcmin * field_size_arcmin

    if isinstance(zsrc, float):
        ngals = int(ngals)
    elif zsrc in ("chang13", "desc_srd"):
        z_distrib_func = zdist.chang2013 if zsrc == "chang13" else zdist.desc_srd
        # Compute the normalisation for the redshift distribution function (z=[0, inf))
        # z_distrib_func(0, is_cdf=True)=0
        norm = z_distrib_func(np.inf, is_cdf=True)
        # Probability to find the galaxy in the requested redshift range
        prob = (
            z_distrib_func(zsrc_max, is_cdf=True) - z_distrib_func(zsrc_min, is_cdf=True)
        ) / norm
        ngals = int(ngals * prob)
    else:
        raise ValueError(f"zsrc (={zsrc}) must be float, 'chang13' or 'desc_srd'")
    return ngals


def _generate_galaxy_catalog(
    cluster_m,
    cluster_z,
    cluster_c,
    cosmo,
    ngals,
    zsrc,
    cluster_ra=None,
    cluster_dec=None,
    delta_so=None,
    massdef=None,
    halo_profile_model=None,
    zsrc_min=None,
    zsrc_max=None,
    shapenoise=None,
    mean_e_err=None,
    photoz_sigma_unscaled=None,
    pz_bins=101,
    pz_quantiles_conf=(5, 31),
    pzpdf_type="shared_bins",
    coordinate_system="euclidean",
    field_size=None,
):
    """A private function that skips the sanity checks on derived properties. This
    function should only be used when called directly from `generate_galaxy_catalog`.
    For a detailed description of each of the parameters, see the documentation of
    `generate_galaxy_catalog`.
    """
    # Too many local variables (22/15)
    # pylint: disable=R0914

    # Set the source galaxy redshifts
    galaxy_catalog = _draw_source_redshifts(zsrc, zsrc_min, zsrc_max, ngals)

    # Add photo-z errors and pdfs to source galaxy redshifts
    if photoz_sigma_unscaled is not None:
        galaxy_catalog.pzpdf_info["type"] = pzpdf_type
        galaxy_catalog = _compute_photoz_pdfs(
            galaxy_catalog,
            photoz_sigma_unscaled,
            pz_bins=pz_bins,
            pz_quantiles_conf=pz_quantiles_conf,
        )

    # Draw galaxy positions
    galaxy_catalog = _draw_galaxy_positions(
        galaxy_catalog, ngals, cluster_ra, cluster_dec, cluster_z, cosmo, field_size
    )

    # Compute the shear on each source galaxy
    gamt = compute_tangential_shear(
        galaxy_catalog["r_mpc"],
        mdelta=cluster_m,
        cdelta=cluster_c,
        z_cluster=cluster_z,
        z_src=galaxy_catalog["ztrue"],
        cosmo=cosmo,
        delta_mdef=delta_so,
        halo_profile_model=halo_profile_model,
        massdef=massdef,
        z_src_info="discrete",
    )

    gamx = np.zeros(ngals)

    kappa = compute_convergence(
        galaxy_catalog["r_mpc"],
        mdelta=cluster_m,
        cdelta=cluster_c,
        z_cluster=cluster_z,
        z_src=galaxy_catalog["ztrue"],
        cosmo=cosmo,
        delta_mdef=delta_so,
        halo_profile_model=halo_profile_model,
        massdef=massdef,
        z_src_info="discrete",
    )

    c_cl = SkyCoord(cluster_ra * u.deg, cluster_dec * u.deg, frame="icrs")
    c_gal = SkyCoord(galaxy_catalog["ra"] * u.deg, galaxy_catalog["dec"] * u.deg, frame="icrs")

    # position angle of drawn galaxies w.r.t cluster center
    _, posangle = c_cl.separation(c_gal).rad, c_cl.position_angle(c_gal).rad
    posangle += 0.5 * np.pi  # for right convention

    # corresponding shear1,2 components
    gam1 = -gamt * np.cos(2 * posangle) + gamx * np.sin(2 * posangle)
    gam2 = -gamt * np.sin(2 * posangle) - gamx * np.cos(2 * posangle)

    # instrinsic ellipticities
    e1_intrinsic = 0
    e2_intrinsic = 0

    # Add shape noise to source galaxy shears
    if shapenoise is not None:
        e1_intrinsic = shapenoise * np.random.standard_normal(ngals)
        e2_intrinsic = shapenoise * np.random.standard_normal(ngals)

    # Compute ellipticities
    galaxy_catalog["e1"], galaxy_catalog["e2"] = compute_lensed_ellipticity(
        e1_intrinsic, e2_intrinsic, gam1, gam2, kappa
    )

    cols = ["ra", "dec", "e1", "e2"]
    # if adding uncertainties
    if mean_e_err is not None:
        galaxy_catalog["e_err"] = mean_e_err * np.random.uniform(0.9, 1.1, ngals)
        galaxy_catalog["e1"] = np.random.normal(galaxy_catalog["e1"], galaxy_catalog["e_err"])
        galaxy_catalog["e2"] = np.random.normal(galaxy_catalog["e2"], galaxy_catalog["e_err"])
        cols += ["e_err"]
    cols += ["z", "ztrue"]
    if all(c is not None for c in (photoz_sigma_unscaled, pzpdf_type)):
        if galaxy_catalog.pzpdf_info["type"] == "individual_bins":
            cols += ["pzbins"]
        if galaxy_catalog.pzpdf_info["type"] == "quantiles":
            cols += ["pzquantiles"]
        else:
            cols += ["pzpdf"]

    if coordinate_system == "celestial":
        galaxy_catalog["e2"] *= -1  # flip e2 to match the celestial coordinate system

    return galaxy_catalog[cols]


def _draw_source_redshifts(zsrc, zsrc_min, zsrc_max, ngals):
    """Set source galaxy redshifts either set to a fixed value or draw from a predefined
    distribution. Return a table (GCData) of the source galaxies

    Uses a sampling technique found in Numerical Recipes in C, Chap 7.2: Transformation Method.
    Pulling out random values from a given probability distribution.

    Parameters
    ----------
    zsrc : float or str
        Choose the source galaxy distribution to be fixed or drawn from a predefined distribution.
        float : All sources galaxies at this fixed redshift.
        str : Draws individual source gal redshifts from predefined distribution. Options are:

            * `chang13` - Chang et al. 2013 (arXiv:1305.0793);
            * `desc_srd` - LSST/DESC Science Requirement Document (arxiv:1809.01669);

    zsrc_min : float
        The minimum source redshift allowed.
    zsrc_max : float, optional
        If source redshifts are drawn, the maximum source redshift
    ngals : float
        Number of galaxies to generate

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Table of true and 'measured' photometric redshifts, which here the same. Redshift
        photometric errors are then added using _compute_photoz_pdfs.

    Notes
    -----
    Much of this code in this function was adapted from the Dallas group
    """
    # Set zsrc to constant value
    if isinstance(zsrc, float):
        zsrc_list = np.ones(ngals) * zsrc

    # Draw zsrc from Chang et al. 2013
    elif zsrc == "chang13":
        zsrc_list = _draw_random_points_from_distribution(
            zsrc_min, zsrc_max, ngals, zdist.chang2013
        )

    # Draw zsrc from the distribution used in the DESC SRD (arxiv:1809.01669)
    elif zsrc == "desc_srd":
        zsrc_list = _draw_random_points_from_distribution(zsrc_min, zsrc_max, ngals, zdist.desc_srd)

    # Draw zsrc from a uniform distribution between zmin and zmax
    elif zsrc == "uniform":
        zsrc_list = np.random.uniform(zsrc_min, zsrc_max, ngals)

    # Invalid entry
    else:
        raise ValueError(f"zsrc must be a float, chang13 or desc_srd. You set: {zsrc}")

    return GCData([zsrc_list, zsrc_list], names=("ztrue", "z"))


def _compute_photoz_pdfs(
    galaxy_catalog, photoz_sigma_unscaled, pz_bins=101, pz_quantiles_conf=(5, 31)
):
    """Private function to add photo-z errors and PDFs to the mock catalog.

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
        Input galaxy catalog to which photoz PDF will be added
    photoz_sigma_unscaled : float
        Width of the Gaussian PDF, without the (1+z) factor
    pz_bins: int, sequence of scalars or str
        Photo-z pdf bins in the given range. If int, the limits are set automatically.
        If is array, must be the bin edges.

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Output galaxy catalog with columns corresponding to the bins
        and values of the redshift PDF for each galaxy.
    """
    galaxy_catalog["pzsigma"] = photoz_sigma_unscaled * (1.0 + galaxy_catalog["ztrue"])
    galaxy_catalog["z"] = galaxy_catalog["ztrue"] + galaxy_catalog[
        "pzsigma"
    ] * np.random.standard_normal(len(galaxy_catalog))

    if galaxy_catalog.pzpdf_info["type"] is None:
        return galaxy_catalog

    zmin = galaxy_catalog["z"] - 10.0 * galaxy_catalog["pzsigma"]
    zmax = galaxy_catalog["z"] + 10.0 * galaxy_catalog["pzsigma"]
    zmin[zmin < 0] = 0.0

    if galaxy_catalog.pzpdf_info["type"] == "shared_bins":
        if isinstance(pz_bins, int):
            galaxy_catalog.pzpdf_info["zbins"] = np.linspace(zmin.min(), zmax.max(), pz_bins)
        else:
            galaxy_catalog.pzpdf_info["zbins"] = np.array(pz_bins)
        galaxy_catalog["pzpdf"] = gaussian(
            galaxy_catalog.pzpdf_info["zbins"],
            galaxy_catalog["z"][:, None],
            galaxy_catalog["pzsigma"][:, None],
        )
    elif galaxy_catalog.pzpdf_info["type"] == "individual_bins":
        if isinstance(pz_bins, int):
            galaxy_catalog["pzbins"] = np.linspace(zmin, zmax, pz_bins).T
        else:
            galaxy_catalog["pzbins"] = [
                pz_bins[max(np.digitize(z1, pz_bins) - 1, 0) : np.digitize(z2, pz_bins) + 1]
                for z1, z2 in zip(zmin, zmax)
            ]
        galaxy_catalog["pzpdf"] = [
            gaussian(row["pzbins"], row["z"], row["pzsigma"]) for row in galaxy_catalog
        ]
    elif galaxy_catalog.pzpdf_info["type"] == "quantiles":
        sigma_steps = np.linspace(-pz_quantiles_conf[0], pz_quantiles_conf[0], pz_quantiles_conf[1])
        galaxy_catalog.pzpdf_info["quantiles"] = 0.5 * erfc(-sigma_steps / np.sqrt(2))
        galaxy_catalog["pzquantiles"] = (
            galaxy_catalog["z"][:, None] + sigma_steps * galaxy_catalog["pzsigma"][:, None]
        )
    else:
        raise ValueError(
            "Value of pzpdf_info['type'] " f"(={galaxy_catalog.pzpdf_info['type']}) " "not valid."
        )
    return galaxy_catalog


def _draw_galaxy_positions(
    galaxy_catalog, ngals, cluster_ra, cluster_dec, cluster_z, cosmo, field_size
):
    """Draw positions of source galaxies around lens

    We draw physical x and y positions from uniform distribution with -4 and 4 Mpc of the
    lensing cluster center. We then convert these to RA and DEC using the supplied cosmology

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
        Source galaxy catalog
    ngals : float
        The number of source galaxies to draw
    cluster_ra : float
        The cluster right ascension in degrees
    cluster_dec : float
        The cluster declination in degrees
    cluster_z : float
        The cluster redshift
    cosmo : dict
        Dictionary of cosmological parameters. Must contain at least, Omega_c, Omega_b,
        and H0
    field_size : float
        The size of the field (field_size x field_size) to be simulated around the cluster center.
        Proper distance in Mpc at the cluster redshift.

    Returns
    -------
    galaxy_catalog : clmm.GCData
        Source galaxy catalog with positions added
    """
    lens_distance = cosmo.eval_da(cluster_z)  # Mpc

    # Draw galaxy around (ra,dec)=(0.0)
    galaxy_catalog["x_mpc"] = np.random.uniform(-(field_size / 2.0), field_size / 2.0, size=ngals)
    galaxy_catalog["y_mpc"] = np.random.uniform(-(field_size / 2.0), field_size / 2.0, size=ngals)
    galaxy_catalog["r_mpc"] = np.sqrt(galaxy_catalog["x_mpc"] ** 2 + galaxy_catalog["y_mpc"] ** 2)

    # ra and dec for a cluster in (0,0)
    ra = -np.rad2deg(galaxy_catalog["x_mpc"] / lens_distance)
    dec = np.rad2deg(galaxy_catalog["y_mpc"] / lens_distance)
    coord_gals = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    coord_cl = SkyCoord(0.0 * u.deg, 0.0 * u.deg, frame="icrs")

    # position angle of drawn galaxies w.r.t cluster center in original position coord_cl = (0,0)
    position_angle = coord_cl.position_angle(coord_gals).to(u.deg)

    # separation of drawn galaxies w.r.t cluster center in original position coord_cl = (0,0)
    sep = coord_cl.separation(coord_gals)

    # cluster actual position
    c2_cl = SkyCoord(cluster_ra * u.deg, cluster_dec * u.deg, frame="icrs")

    # new galaxy (ra,dec) w.r.t the new cluster position c2_cl
    new_coord = c2_cl.directional_offset_by(position_angle, sep)

    galaxy_catalog["ra"] = new_coord.ra.degree
    galaxy_catalog["dec"] = new_coord.dec.degree

    return galaxy_catalog


def _find_aphysical_galaxies(galaxy_catalog, zsrc_min):
    r"""Finds the galaxies that have aphysical derived values due to large systematic choices.

    Currently checks the following conditions
    e1 \in [-1, 1]
    e2 \in [-1, 1]
    z  < zsrc_min
    This was converted to a seperate function to allow for ease of extension without needing
    to change the same code in multiple locations.

    Parameters
    ----------
    galaxy_catalog : clmm.GCData
        Galaxy source catalog
    zsrc_min : float
        Minimum galaxy redshift allowed

    Returns
    -------
    nbad : int
        The number of aphysical galaxies in galaxy_catalog
    badgals : array_like
        A list of the indicies in galaxy_catalog that need to be redrawn
    """
    etot = np.sqrt(
        galaxy_catalog["e1"] * galaxy_catalog["e1"] + galaxy_catalog["e2"] * galaxy_catalog["e2"]
    )

    #     badgals = np.where((np.abs(galaxy_catalog['e1']) > 1.0) |
    #                        (np.abs(galaxy_catalog['e2']) > 1.0) |
    #                        (galaxy_catalog['ztrue'] < zsrc_min)
    #                       )[0]

    badgals = np.where((etot > 1.0) | (galaxy_catalog["ztrue"] < zsrc_min))[0]
    nbad = len(badgals)
    return nbad, badgals
