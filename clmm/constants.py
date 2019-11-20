""" Provide a consistent set of constants to use through CLMM """
import astropy.constants as astropyconst
import astropy.units as u

# In modeling.py, get_critical_surface_density

# Speed of light
# Units: km/s
# Source: Astropy - CODATA 2014
CLIGHT_KMS = astropyconst.c.to(u.km/u.s).value

# Newton's constant in m^3/kg/s^2
# Units: m^3/kg/s^2
# Source: Astropy - CODATA 2014
GNEWT = astropyconst.G.to(u.m**3/u.kg/u.s**2)
