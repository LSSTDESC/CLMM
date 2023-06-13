"""General utility functions that are used in multiple modules"""
from astropy import units as u


def convert_units(dist1, unit1, unit2, redshift=None, cosmo=None):
    """Convenience wrapper to convert between a combination of angular and physical units.

    Supported units: radians, degrees, arcmin, arcsec, Mpc, kpc, pc
    (letter case independent)

    To convert between angular and physical units you must provide both
    `redshift` and a CLMM Cosmology object `cosmo`.

    Parameters
    ----------
    dist1 : float, array_like
        Input distances
    unit1 : str
        Unit for the input distances
    unit2 : str
        Unit for the output distances
    redshift : float, None, optional
        Redshift used to convert between angular and physical units. Default: None
    cosmo : clmm.Cosmology, None, optional
        CLMM Cosmology object to compute angular diameter distance to
        convert between physical and angular units. Default: None

    Returns
    -------
    dist2: float, numpy.ndarray
        Input distances converted to unit2
    """
    # make case independent
    unit1, unit2 = unit1.lower(), unit2.lower()
    # Available units
    angular_bank = {
        "radians": u.rad,
        "degrees": u.deg,
        "arcmin": u.arcmin,
        "arcsec": u.arcsec,
    }
    physical_bank = {"pc": u.pc, "kpc": u.kpc, "mpc": u.Mpc}
    units_bank = {**angular_bank, **physical_bank}
    # Some error checking
    if unit1 not in units_bank:
        raise ValueError(f"Input units ({unit1}) not supported")
    if unit2 not in units_bank:
        raise ValueError(f"Output units ({unit2}) not supported")
    # Try automated astropy unit conversion
    try:
        dist2 = (dist1 * units_bank[unit1]).to(units_bank[unit2]).value
    # Otherwise do manual conversion
    except u.UnitConversionError:
        # Make sure that we were passed a redshift and cosmology
        if redshift is None or cosmo is None:
            raise TypeError(
                "Redshift and cosmology must be specified to convert units"
            ) from u.UnitConversionError
        # Redshift must be greater than zero for this approx
        if not redshift > 0.0:
            raise ValueError("Redshift must be greater than 0.") from u.UnitConversionError
        # Convert angular to physical
        if (unit1 in angular_bank) and (unit2 in physical_bank):
            dist1_rad = (dist1 * units_bank[unit1]).to(u.rad).value
            dist1_mpc = cosmo.rad2mpc(dist1_rad, redshift)
            dist2 = (dist1_mpc * u.Mpc).to(units_bank[unit2]).value
        # Otherwise physical to angular
        else:
            dist1_mpc = (dist1 * units_bank[unit1]).to(u.Mpc).value
            dist1_rad = cosmo.mpc2rad(dist1_mpc, redshift)
            dist2 = (dist1_rad * u.rad).to(units_bank[unit2]).value
    return dist2
