"""General utility functions that are used in multiple modules"""

from .beta_lens import (
    compute_beta,
    compute_beta_s,
    compute_beta_s_func,
    compute_beta_mean,
    compute_beta_s_mean,
    compute_beta_s_square_mean,
)

from .boost import (
    compute_nfw_boost,
    compute_powerlaw_boost,
    correct_sigma_with_boost_values,
    correct_sigma_with_boost_model,
    boost_models,
)

from .ellipticity import (
    convert_shapes_to_epsilon,
    build_ellipticities,
    compute_lensed_ellipticity,
)

from .statistic import (
    compute_weighted_bin_sum,
    compute_radial_averages,
    make_bins,
    _draw_random_points_from_distribution,
    _draw_random_points_from_tab_distribution,
    gaussian,
)

from .validation import (
    arguments_consistency,
    _patch_rho_crit_to_cd2018,
    validate_argument,
    _validate_ra,
    _validate_dec,
    _validate_is_deltasigma_sigma_c,
)

from .units import (
    convert_units,
)
