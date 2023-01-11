

## DEMONSTRATES HOW TO CONVERT THE MASS AND CONCENTRATION BETWEEN VARIOUS MASS DEFINITIONS (FROM 200m TO 500c IN THIS
## EXAMPLE), AND RELATED FUNCTIONALITIES.


##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||||||||||||||||||||| SETUP ||||||||||||||||||||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

import os
os.environ['CLMM_MODELING_BACKEND'] = 'ccl'
import warnings
warnings.filterwarnings('ignore', message='.*(!).*')
import clmm
import numpy as np
import matplotlib.pyplot as plt


## DEFINE THE COSMOLOGY
H0 = 70.
Omega_b0 = 0.045
Omega_dm0 = 0.27 - Omega_b0
Omega_k0 = 0.
cosmo = clmm.Cosmology(H0=H0, Omega_dm0=Omega_dm0, Omega_b0=Omega_b0, Omega_k0=Omega_k0)


## DEFINE THE HALO PARAMETERS FIRST USING THE 200m OVERDENSITY DEFINITION WITH:
##      1) MASS M_200m
##      2) CONCENTRATION c_200m
M1 = 1.e14
c1 = 3
massdef1 = 'mean'
delta_mdef1 = 200
z_cl = 0.4


## CREATE A clmm MODELING OBJECT FOR EACH PROFILE PARAMETRIZATION
nfw_def1 = clmm.Modeling(massdef=massdef1, delta_mdef=delta_mdef1, halo_profile_model='nfw')
her_def1 = clmm.Modeling(massdef=massdef1, delta_mdef=delta_mdef1, halo_profile_model='hernquist')
ein_def1 = clmm.Modeling(massdef=massdef1, delta_mdef=delta_mdef1, halo_profile_model='einasto')

## SET THE PROPERTIES OF THE PROFILES
nfw_def1.set_mass(M1)
nfw_def1.set_concentration(c1)
nfw_def1.set_cosmo(cosmo)

her_def1.set_mass(M1)
her_def1.set_concentration(c1)
her_def1.set_cosmo(cosmo)

ein_def1.set_mass(M1)
ein_def1.set_concentration(c1)
ein_def1.set_cosmo(cosmo)



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##||||||||||||||||||||||||||||||||||| COMPUTE THE ENCLOSED MASS IN A GIVEN RADIUS |||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

## CALCULATE THE ENCLOSED MASSES WITHIN r WITH THE CLASS METHOD eval_mass_in_radius.
## THE CALCULATION CAN ALSO BE DONE IN THE FUNCTIONAL INTERFACE WITH compute_profile_mass_in_radius.
## THE ENCLOSED MASS IS CALCULATED AS
##          M(<r) = M_{\Delta} f(c_{\Delta} r/r_{\Delta}) / f(c_{\Delta})
## WHERE f(x) IS OF THE FORM:
##      @ NFW: f(x) = ln(1+x) - x/(1+x)
##      @ EINASTO: f(x) = \gamma(3/\alpha, 2x^\alpha/\alpha)    [\gamma is the lower incomplete gamma function]
##      @ HERQUIST: f(x) = (x/(1+x))^2

r = np.logspace(-2, 0.4, 100)
nfw_def1_enclosed = nfw_def1.eval_mass_in_radius(r3d=r, z_cl=z_cl)
her_def1_enclosed = her_def1.eval_mass_in_radius(r3d=r, z_cl=z_cl)
ein_def1_enclosed = ein_def1.eval_mass_in_radius(r3d=r, z_cl=z_cl)


## CHECK OUT THE ENCLOSED MASS BETWEEN THE DIFFERENT PROFILE MODELS
fig = plt.figure(figsize=(8,6))
fig.gca().loglog(r, nfw_def1_enclosed, label='NFW')
fig.gca().loglog(r, her_def1_enclosed, label='Hernquist')
fig.gca().loglog(r, ein_def1_enclosed, label='Einasto')
fig.gca().set_xlabel(r'$r$ [Mpc]', fontsize=14)
fig.gca().set_ylabel(r'$M(<r)$ [$M_\odot$]', fontsize=14)
fig.gca().legend()
plt.tight_layout()
plt.show()



##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
##|||||||||||||||||||||||||||||||||||| COMPUTE THE SPHERICAL OVERDENSITY RADIUS |||||||||||||||||||||||||||||||||||||
##VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
