""" Provide a consistent set of constants to use through CLMM """
import astropy.constants as astropyconst
import astropy.units as u

# In modeling.py, get_critical_surface_density
# Add: from .constants import CLIGHT_KMS, GNEWT, PC_TO_METER, SOLAR_MASS
# Revise: G = GNEWT * SOLAR_MASS / PC_TO_METER**3
# Revise: c = CLIGHT_KMS * 1000. / PC_TO_METER

# Speed of light
# Units: km/s
# Source: Astropy - CODATA 2014
# Same source as CCL, revise to CCL when available
CLIGHT_KMS = astropyconst.c.to(u.km/u.s).value

# Newton's constant in m^3/kg/s^2
# Units: m^3/kg/s^2
# Source: Astropy - CODATA 2014
# CCL Does not provide source, revise to CCL when available
GNEWT = astropyconst.G.to(u.m**3/u.kg/u.s**2).value

# parsec to meter
# Units: m
# Source: PDG 2013
# Same source as CCL, revise to CCL when available
PC_TO_METER = 3.08567758149e16

# Solar mass
# Units: kg
# Source: GSL
# Same source as CCL, revise to CCL when available
SOLAR_MASS = 1.98892e30
