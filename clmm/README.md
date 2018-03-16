# `clmm` proposed code structure

`clmm` is a general code for performing individual- and population-level inference on galaxy cluster weak lensing data.  `clmm` aims to be modular in (at least) three respects:

1. `clmm` will be able to run on real data as well as simulations, and it will not be restricted to any particular datasets.
2. `clmm` will support multiple modes of inference of the cluster mass function and other relevant distributions, such as the mass-concentration relation.
3. `clmm` will enable evaluation of results on the basis of a number of different metrics, some of which will not require a notion of truth from a simulation.

To enable these goals, `clmm` will have (at least) three major superclasses of objects:

0. `clmm.utils`

1. `clmm.Reader` objects that are able to load data and corresponding metadata from external files; subclass instances will be specific to the native format of the pipeline that produced the data and/or the type of data (shear maps vs. mass surface density maps, etc.)
 A. config file readers for `clmm.Halo.map`, `clmm.Halo.mass`, etc. reading in data itself in different original formats from simulations/observations 

2. `clmm.Halo` class
 Z. `clmm.Halo.config_params`
 A. `clmm.`from `clmm.Halo.Map` superclass (???)
  i. `clmm.Halo.map.data['shear']` and `clmm.Halo.map.data['convergence']`
  ii. `clmm.Halo.map.metadata` including radial range, binning, miscentering, noise level for a profile fit, e.g.
  iii. `clmm.Halo.map.convert('mass2D', 'shear', config_params)` (make new maps i.e. `clmm.Halo.map.data['shear']` from existing maps i.e.  `clmm.Halo.map.data['mass2D']`)
 B. `clmm.Halo.redshift` 
 C. `clmm.Halo.mass`
  i. `clmm.Halo.mass.MLE`
  ii. `clmm.Halo.mass.true_3D`
 D. `clmm.Halo.utils`
  i. `clmm.Halo.utils.cross_correlator()` taking two maps and producing another map
 E. `clmm.Halo.build_profile(clmm.Halo.map)` requiring `clmm.Halo.config_params` and checking if unspecified, producing `clmm.Halo.obs_profile` 
  i. `clmm.Halo.obs_profile` as an instance of the `clmm.Halo.Profile` class

3. `clmm.Halo_Model` superclass, must have `.evaluate_func()` (like the NFW functional form) and `.evaluate_prior()` (such as uniform distribution or Gaussian, etc.)
 A. consider `clmm.NFW` subclass taking config file with prior params
  i. `clmm.NFW.evaluate_function(x, model_params)` where data can be `clmm.Halo.profile` and params is list(?), evaluates a function
  ii. `clmm.NFW.evaluate_likelihood((xobs, yobs), model_params)` which calls `clmm.NFW.evaluate_function()`
  ii. `clmm.NFW.evaluate_prior(model_params)` evaluates p(model\_params | prior\_config\_params)
 B. mass aperture has different data, etc.

4. `clmm.Inferrer` objects that accept data produced by the `clmm.Reader` subclass object as well as other information, such as theoretical models and selection/systematics functions, and output estimates of the desired quantity, such as a probability distribution over the mass-concentration relation or a stacked estimator of the mass function.
 A. `clmm.Halo_Inferrer(clmm.Halo, config_stuff, **params)` class
  i. produces output in the form of posteriors (or point estimates)
  ii. `clmm.Halo_Inferrer.profile_fitter(clmm.Halo.map, profile_config_params)` adding fit info to `clmm.Halo.profile`, but maybe later because not well defined right now
  iii. `clmm.Halo_Inferrer(clmm.Halo, clmm.Model, config_params)` outputting posterior (or samples or MLE) for one halo given config\_params for inferrer

5. `clmm.Metrics` objects that take as input one or more estimators produced by `clmm.Inferrer` objects and, when available, the true value of the estimated quantity read in by a `clmm.Reader` object separately.

6. extensions for many-halo inference __punting to later__
 A. `clmm.Halo_Ensemble` 
  i. `clmm.Halo_Ensemble.select()` that bins `clmm.Halo` objects by some attribute they have
 B. `clmm.Halo_Ensemble_Inferrer(clmm.Halo_Ensemble, clmm.)`

Users and developers may implement their own subclass instances of these superclasses to flesh out the `clmm` functionality.
