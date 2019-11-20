******************************
Polar Averaging
******************************

Most cluster weak lensing mass measurements use polar averaging
(profile making) of the data before fitting the data to a model
profile, such as NFW and Einasto profiles. This repository contains
tools to create weak lensing profiles from a dataset.

Polar Averaged Mock Data
========================

We first ......

.. math::
   
   \rho_{\rm nfw}(r) = \frac{\Omega_m\rho_{\rm crit}\delta_c}{\left(\frac{r}{r_s}\right)\left(1+\frac{r}{r_s}\right)^2}.

The free parameters are the cluster mass :math:`M_\Delta` and concentration :math:`c_\Delta = r_\Delta/r_s`. In this module we choose to define the density with respect to the matter background density :math:`\Omega_m\rho_{\rm crit}`. The scale radius :math:`r_s` is given in :math:`h^{-1}{\rm Mpc}`, however the code uses concentration :math:`c_\Delta` as an argument instead. The normalization :math:`\delta_c` is calculated internally and depends only on the concentration. As written, because of the choice of units the only cosmological parameter that needs to be passed in is :math:`\Omega_m`.

.. note::
   The density profiles can use :math:`\Delta\neq 200`.

To use this, you would do:

.. code::

   from cluster_toolkit import density
   import numpy as np
   radii = np.logspace(-2, 3, 100) #Mpc/h comoving
   mass = 1e14 #Msun/h
   concentration = 5 #arbitrary
   Omega_m = 0.3
   rho_nfw = density.rho_nfw_at_r(radii, mass, concentration, Omega_m)


Einasto Profile
===============

The `Einasto profile <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`_ is a 3D density profile given by:

.. math::
   
   \rho_{\rm ein}(r) = \rho_s\exp\left(-\frac{2}{\alpha}\left(\frac{r}{r_s}\right)^\alpha\right)

In this model, the free parameters are the scale radius :math:`r_s`, :math:`\alpha`, and the cluster mass :math:`M_\Delta`. The scale density :math:`\rho_s` is calculated internally, or can be passed in instead of mass. To use this, you would do:

.. code::

   from cluster_toolkit import density
   import numpy as np
   radii = np.logspace(-2, 3, 100) #Mpc/h comoving
   mass = 1e14 #Msun/h
   r_scale = 1.0 #Mpc/h comoving scale radius
   alpha = 0.19 #arbitrary; a typical value
   Omega_m = 0.3
   rho_ein = density.rho_einasto_at_r(radii, mass, r_scale, alpha, Omega_m)

We can see the difference between these two profiles here:

.. image:: figures/density_profile_example.png
