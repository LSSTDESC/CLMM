"""@file miscentering.py
Model functions with miscentering
"""
# Functions to model halo profiles

import numpy as np
from scipy.integrate import quad, dblquad, tplquad

def integrand_surface_density_nfw(theta, r_proj, r_mis, r_s):
    x = np.sqrt(r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta)) / r_s

    x2m1 = x**2.0 - 1.0
    if x < 1:
        sqrt_x2m1 = np.sqrt(-x2m1)
        res = np.arcsinh(sqrt_x2m1 / x) / (-x2m1) ** (3.0 / 2.0) + 1.0 / x2m1
    elif x > 1:
        sqrt_x2m1 = np.sqrt(x2m1)
        res = -np.arcsin(sqrt_x2m1 / x) / (x2m1) ** (3.0 / 2.0) + 1.0 / x2m1
    else:
        res = 1.0 / 3.0
    return res

def integrand_surface_density_einasto(r_par, theta, r_proj, r_mis, r_s, alpha_ein):
    # Projected surface mass density element for numerical integration
    x = (
        np.sqrt(
            r_par**2.0 + r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta)
        )
        / r_s
    )

    return np.exp(-2.0 * (x**alpha_ein - 1.0) / alpha_ein)

def integrand_surface_density_hernquist(theta, r_proj, r_mis, r_s):
    x = np.sqrt(r_proj**2.0 + r_mis**2.0 - 2.0 * r_proj * r_mis * np.cos(theta)) / r_s

    x2m1 = x**2.0 - 1.0
    if x < 1:
        res = -3 / x2m1**2 + (x2m1 + 3) * np.arcsinh(np.sqrt(-x2m1) / x) / (-x2m1) ** 2.5
    elif x > 1:
        res = -3 / x2m1**2 + (x2m1 + 3) * np.arcsin(np.sqrt(x2m1) / x) / (x2m1) ** 2.5
    else:
        res = 4.0 / 15.0
    return res

def integrand_mean_surface_density_nfw(theta, r_proj, r_mis, r_s):
    return r_proj * integrand_surface_density_nfw(theta, r_proj, r_mis, r_s)

def integrand_mean_surface_density_einasto(z, theta, r_proj, r_mis, r_s, alpha_ein):
    return r_proj * integrand_surface_density_einasto(
        z, theta, r_proj, r_mis, r_s, alpha_ein
    )

def integrand_mean_surface_density_hernquist(theta, r_proj, r_mis, r_s):
    return r_proj * integrand_surface_density_hernquist(theta, r_proj, r_mis, r_s)

def eval_surface_density(r_proj, z_cl, r_mis, integrand, norm, aux_args, extra_integral):
    args_list = [(r, r_mis, *aux_args) for r in r_proj]
    if extra_integral:
        res = [
            dblquad(integrand, 0.0, np.pi, 0, np.inf, args=args, epsrel=1e-6)[0]
            for args in args_list
        ]
    else:
        res = [quad(integrand, 0.0, np.pi, args=args, epsrel=1e-6)[0] for args in args_list]

    res = np.array(res) * norm / np.pi
    return res

def eval_mean_surface_density(r_proj, z_cl, r_mis, integrand, norm, aux_args, extra_integral):

    r_lower = np.zeros_like(r_proj)
    r_lower[1:] = r_proj[:-1]
    args = (r_mis, *aux_args)

    if extra_integral:
        res = [
            tplquad(integrand, r_low, r_high, 0, np.pi, 0, np.inf, args=args, epsrel=1e-6)[0]
            for r_low, r_high in zip(r_lower, r_proj)
        ]
    else:
        res = [
            dblquad(integrand, r_low, r_high, 0, np.pi, args=args, epsrel=1e-6)[0]
            for r_low, r_high in zip(r_lower, r_proj)
        ]

    return np.cumsum(res) * norm * 2 / np.pi / r_proj**2
