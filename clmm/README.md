# `clmm` proposed code structure

`clmm` is a general code for performing individual- and population-level inference on galaxy cluster weak lensing data.  `clmm` aims to be modular in (at least) three respects:

1. `clmm` will be able to run on real data as well as simulations, and it will not be restricted to any particular datasets.
2. `clmm` will support multiple modes of inference of the cluster mass function and other relevant distributions, such as the mass-concentration relation.
3. `clmm` will enable evaluation of results on the basis of a number of different metrics, some of which will not require a notion of truth from a simulation.

To enable these goals, `clmm` will have (at least) three major superclasses of objects:

0. `clmm.utils`
1. `clmm.Reader` objects that are able to load data and corresponding metadata from external files; subclass instances will be specific to the native format of the pipeline that produced the data and/or the type of data (shear maps vs. mass surface density maps, etc.)
A. config file readers for `clmm.Halo.map`, `clmm.Halo.mass`, etc. reading in data itself
B. `clmm.Inferrer` config file reader including profile/prior/fit method
2. `clmm.Halo` class
A. `clmm.Halo.map` class
i. `clmm.Halo.map.shear`
ii. `clmm.Halo.map.convergence`
iii. `clmm.Halo.map.mass.convert()` (make new maps from existing maps)
B. `clmm.Halo.redshift` 
C. `clmm.Halo.mass`
i. `clmm.Halo.mass.MLE`
ii. `clmm.Halo.mass.true_3D`
D. `clmm.Halo.utils`
i. `clmm.Halo.utils.cross_correlator()` taking two maps and producing another map
ii. `clmm.Halo.utils.`
E. `clmm.Halo.preprocess()`
3. `clmm.Selector` that bins `clmm.Halo` objects by some attribute they have
4. `clmm.Inferrer` objects that accept data produced by the `clmm.Reader` subclass object as well as other information, such as theoretical models and selection/systematics functions, and output estimates of the desired quantity, such as a probability distribution over the mass-concentration relation or a stacked estimator of the mass function.
A. `clmm.Single_Inferrer(clmm.Halo.map, **params)`
i. `params` including radial range, binning, miscentering, noise level for a profile fit, e.g.
ii. `clmm.Single_Inferrer.profile_fitter(clmm.Halo.map, profile_config_params)` adding fit info to `clmm.Halo`
B. `clmm.Ensemble_Inferrer(clmm.Ensemble)`
5. `clmm.Metrics` objects that take as input one or more estimators produced by `clmm.Inferrer` objects and, when available, the true value of the estimated quantity read in by a `clmm.Reader` object separately.

Users and developers may implement their own subclass instances of these superclasses to flesh out the `clmm` functionality.
 
