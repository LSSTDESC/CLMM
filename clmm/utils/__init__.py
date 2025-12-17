"""General utility functions that are used in multiple modules"""

from . import constants, plotting, redshift_distributions
from .beta_lens import (
    compute_beta,
    compute_beta_s,
    compute_beta_s_func,
    compute_beta_s_mean_from_distribution,
    compute_beta_s_mean_from_weights,
    compute_beta_s_square_mean_from_distribution,
    compute_beta_s_square_mean_from_weights,
)
from .boost import (
    boost_models,
    compute_nfw_boost,
    compute_powerlaw_boost,
    correct_with_boost_model,
    correct_with_boost_values,
)
from .ellipticity import (
    build_ellipticities,
    compute_lensed_ellipticity,
    convert_shapes_to_epsilon,
)
from .redshift_tools import _integ_pzfuncs, compute_for_good_redshifts
from .statistic import (
    _draw_random_points_from_distribution,
    _draw_random_points_from_tab_distribution,
    compute_radial_averages,
    compute_weighted_bin_sum,
    gaussian,
    make_bins,
)
from .units import convert_units
from .validation import (
    DiffArray,
    _patch_rho_crit_to_cd2018,
    _validate_coordinate_system,
    _validate_dec,
    _validate_is_deltasigma_sigma_c,
    _validate_ra,
    arguments_consistency,
    validate_argument,
)
