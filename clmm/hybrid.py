"""Functions requiring both data and model assumptions"""
# function pointer instead of keywords as cosmo.sigma_crit_filter
# compute tangential and cross in util and appears in calls for polaraveraging
#
# most polaraveraging already in utils
#
# check the cosmology again and again or ask for it to be input each time
# GC has a cosmo, if it has a cosmo, must call from
#
# agnostic version in utils, can call directly if user specifies everything or call from 3 modules with our objects
# base versions in util, dataoperations in

def _convert_rad_to_mpc(dist1, redshift, cosmo, do_inverse=False):
    r""" Convert between radians and Mpc using the small angle approximation
    and :math:`d = D_A \theta`.

    Parameters
    ==========
    dist1 : array_like
        Input distances
    redshift : float
        Redshift used to convert between angular and physical units
    cosmo : astropy.cosmology
        Astropy cosmology object to compute angular diameter distance to
        convert between physical and angular units
    do_inverse : bool
        If true, converts Mpc to radians

    Returns
    =======
    dist2 : array_like
        Converted distances
    """
    d_a = cosmo.angular_diameter_distance(redshift).to('Mpc').value
    if do_inverse:
        return dist1 / d_a
    return dist1 * d_a

def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """ Convenience wrapper to convert between a combination of angular and physical units.

    Supported units: radians, degrees, arcmin, arcsec, Mpc, kpc, pc

    To convert between angular and physical units you must provide both
    a redshift and a cosmology object.

    Parameters
    ----------
    dist1 : array_like
        Input distances
    unit1 : str
        Unit for the input distances
    unit2 : str
        Unit for the output distances
    redshift : float
        Redshift used to convert between angular and physical units
    cosmo : astropy.cosmology
        Astropy cosmology object to compute angular diameter distance to
        convert between physical and angular units

    Returns
    -------
    dist2: array_like
        Input distances converted to unit2
    """
    angular_bank = {"radians": u.rad, "degrees": u.deg, "arcmin": u.arcmin, "arcsec": u.arcsec}
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "Mpc": u.Mpc}
    units_bank = {**angular_bank, **physical_bank}

    # Some error checking
    if unit1 not in units_bank:
        raise ValueError(f"Input units ({unit1}) not supported")
    if unit2 not in units_bank:
        raise ValueError(f"Output units ({unit2}) not supported")

    # Try automated astropy unit conversion
    try:
        return (dist1 * units_bank[unit1]).to(units_bank[unit2]).value

    # Otherwise do manual conversion
    except u.UnitConversionError:
        # Make sure that we were passed a redshift and cosmology
        if redshift is None or cosmo is None:
            raise TypeError("Redshift and cosmology must be specified to convert units")

        # Redshift must be greater than zero for this approx
        if not redshift > 0.0:
            raise ValueError("Redshift must be greater than 0.")

        # Convert angular to physical
        if (unit1 in angular_bank) and (unit2 in physical_bank):
            dist1_rad = (dist1 * units_bank[unit1]).to(u.rad).value
            dist1_mpc = _convert_rad_to_mpc(dist1_rad, redshift, cosmo, do_inverse=False)
            return (dist1_mpc * u.Mpc).to(units_bank[unit2]).value

        # Otherwise physical to angular
        dist1_mpc = (dist1 * units_bank[unit1]).to(u.Mpc).value
        dist1_rad = _convert_rad_to_mpc(dist1_mpc, redshift, cosmo, do_inverse=True)
        return (dist1_rad * u.rad).to(units_bank[unit2]).value


def make_binned_profile(cluster,
                       angsep_units, bin_units, bins=10, cosmo=None,
                       tan_component_in='et', cross_component_in='ex',
                       tan_component_out='gt', cross_component_out='gx', table_name='profile',
                       add_to_cluster=True, include_empty_bins=False, gal_ids_in_bins=False):
    r"""Compute the shear or ellipticity profile of the cluster

    We assume that the cluster object contains information on the cross and
    tangential shears or ellipticities and angular separation of the source galaxies

    This function can be called in two ways using an instance of GalaxyCluster

    1. Pass an instance of GalaxyCluster into the function::

        make_shear_profile(cluster, 'radians', 'radians')

    2. Call it as a method of a GalaxyCluster instance::

        cluster.make_shear_profile('radians', 'radians')

    Parameters
    ----------
    cluster : GalaxyCluster
        Instance of GalaxyCluster that contains the cross and tangential shears or ellipticities of
        each source galaxy in its `galcat`
    angsep_units : str
        Units of the calculated separation of the source galaxies
        Allowed Options = ["radians"]
    bin_units : str
        Units to use for the radial bins of the shear profile
        Allowed Options = ["radians", deg", "arcmin", "arcsec", kpc", "Mpc"]
    bins : array_like, optional
        User defined bins to use for the shear profile. If a list is provided, use that as
        the bin edges. If a scalar is provided, create that many equally spaced bins between
        the minimum and maximum angular separations in bin_units. If nothing is provided,
        default to 10 equally spaced bins.
    cosmo: dict, optional
        Cosmology parameters to convert angular separations to physical distances
    tan_component_in: string, optional
        Name of the column in the `galcat` astropy table of the `cluster` object that contains
        the tangential component to be binned. Default: 'et'
    tan_component_out: string, optional
        Name of the column in the `profile` table of the `cluster` object that will contain
        the binned profile of the tangential component. Default: 'gx'
    cross_component_in: string, optional
        Name of the column in the `galcat` astropy table of the `cluster` object that contains
        the cross component to be binned. Default: 'ex'
    cross_component_out: string, optional
        Name of the column in the `profile` table of the `cluster` object that will contain
        the  binned profile of the cross component. Default: 'gx'
    add_to_cluster: bool, optional
        Attach the profile to the cluster object as `cluster.profile`
    include_empty_bins: bool, optional
        Also include empty bins in the returned table
    gal_ids_in_bins: bool, optional
        Also include the list of galaxies ID belonging to each bin in the returned table

    Returns
    -------
    profile : GCData
        Output table containing the radius grid points, the tangential and cross shear profiles
        on that grid, and the errors in the two shear profiles. The errors are defined as the
        standard errors in each bin.

    Notes
    -----
    This is an example of a place where the cosmology-dependence can be sequestered to another module.
    """
    if not all([t_ in cluster.galcat.columns for t_ in (tan_component_in, cross_component_in, 'theta')]):
        raise TypeError('Shear or ellipticity information is missing!  Galaxy catalog must have tangential' +\
                        'and cross shears (gt,gx) or ellipticities (et,ex). Run compute_tangential_and_cross_components first.')
    if 'z' not in cluster.galcat.columns:
        raise TypeError('Missing galaxy redshifts!')

    # Check to see if we need to do a unit conversion
    if angsep_units is not bin_units:
        source_seps = convert_units(cluster.galcat['theta'], angsep_units, bin_units,
                                    redshift=cluster.z, cosmo=cosmo)
    else:
        source_seps = cluster.galcat['theta']

    # Make bins if they are not provided
    if not hasattr(bins, '__len__'):
        bins = make_bins(np.min(source_seps), np.max(source_seps), bins)

    # Compute the binned averages and associated errors
    r_avg, gt_avg, gt_err, nsrc, binnumber = compute_radial_averages(
        source_seps, cluster.galcat[tan_component_in].data, xbins=bins, error_model='std/sqrt_n')
    r_avg, gx_avg, gx_err, _, _ = compute_radial_averages(
        source_seps, cluster.galcat[cross_component_in].data, xbins=bins, error_model='std/sqrt_n')
    r_avg, z_avg, z_err, _, _ = compute_radial_averages(
        source_seps, cluster.galcat['z'].data, xbins=bins, error_model='std/sqrt_n')

    # Make out table
    profile_table = GCData([bins[:-1], r_avg, bins[1:], gt_avg, gt_err, gx_avg, gx_err,
                            z_avg, z_err, nsrc],
                            names=('radius_min', 'radius', 'radius_max',
                                   tan_component_out, tan_component_out+'_err',
                                   cross_component_out, cross_component_out+'_err',
                                   'z', 'z_err', 'n_src'),
                            meta={'cosmo':cosmo, 'bin_units':bin_units}, # Add metadata
                            )
    # add galaxy IDs
    if gal_ids_in_bins:
        if 'id' not in cluster.galcat.columns:
            raise TypeError('Missing galaxy IDs!')
        profile_table['gal_id'] = [list(cluster.galcat['id'][binnumber==i+1])
                                    for i in np.arange(len(r_avg))]

    # return empty bins?
    if not include_empty_bins:
        profile_table = profile_table[profile_table['n_src'] > 1]

    if add_to_cluster:
        setattr(cluster, table_name, profile_table)

    return profile_table
