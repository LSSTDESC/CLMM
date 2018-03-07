# `clmm` proposed code structure

`clmm` is a general code for performing individual- and population-level inference on galaxy cluster weak lensing data.  `clmm` aims to be modular in (at least) three respects:

1. `clmm` will be able to run on real data as well as simulations, and it will not be restricted to any particular datasets.
2. `clmm` will support multiple modes of inference of the cluster mass function and other relevant distributions, such as the mass-concentration relation.
3. `clmm` will enable evaluation of results on the basis of a number of different metrics, some of which will not require a notion of truth from a simulation.

To enable these goals, `clmm` will have (at least) three major superclasses of objects:

1. `clmm.Reader` objects that are able to load data and corresponding metadata from external files; subclass instances will be specific to the native format of the pipeline that produced the data and/or the type of data (shear maps vs. mass surface density maps, etc.)
2. `clmm.Inferrer` objects that accept data produced by the `clmm.Reader` subclass object as well as other information, such as theoretical models and selection/systematics functions, and output estimates of the desired quantity, such as a probability distribution over the mass-concentration relation or a stacked estimator of the mass function.
3. `clmm.Metrics` objects that take as input one or more estimators produced by `clmm.Inferrer` objects and, when available, the true value of the estimated quantity read in by a `clmm.Reader` object separately.

Users and developers may implement their own subclass instances of these superclasses to flesh out the `clmm` functionality.
