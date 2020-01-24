""" Provide a consistent set of constants to use through CLMM """
from enum import Enum
import astropy.constants as astropyconst
import astropy.units as u


class Constants(Enum):
    """ A set of constants for consistency throughout the
    code and dependencies. """


    CLIGHT_KMS = astropyconst.c.to(u.km/u.s).value
    """ Speed of light (km/s)

    Source: Astropy - CODATA 2014

    Same source as CCL, revise to CCL when available
    """

    GNEWT = astropyconst.G.to(u.m**3/u.kg/u.s**2).value
    """ Newton's constant (m^3/kg/s^2)

    Source: Astropy - CODATA 2014

    CCL Does not provide source, revise to CCL when available
    """

    PC_TO_METER = 3.08567758149e16
    """ parsec to meter (m)

    Source: PDG 2013

    Same source as CCL, revise to CCL when available
    """

    SOLAR_MASS = 1.98892e30
    """ Solar mass (kg)

    Source: GSL

    Same source as CCL, revise to CCL when available
    """
