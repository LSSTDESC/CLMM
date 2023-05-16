""" Provide a consistent set of constants to use through CLMM """
from enum import Enum


class Constants(Enum):
    """A set of constants for consistency throughout the
    code and dependencies."""

    CLIGHT = 299792458.0
    """ Speed of light (m/s)

    Source: CODATA 2018
    """

    CLIGHT_KMS = CLIGHT * 1.0e-3
    """ Speed of light (km/s)

    Source: CODATA 2018
    """

    GNEWT = 6.67430e-11
    """ Newton's constant (m^3/kg/s^2)

    Source: CODATA 2018
    """

    PC_TO_METER = 3.085677581491367e16
    """ parsec to meter (m)

    Source: IAU 2015
    """

    GNEWT_SOLAR_MASS = 1.3271244e20
    """ G x Solar mass (m^3/s^2)

    Source: IAU 2015
    """

    SOLAR_MASS = GNEWT_SOLAR_MASS / GNEWT
    """ Solar mass (kg)

    Source: IAU 2015/CODATA 2018
    """
