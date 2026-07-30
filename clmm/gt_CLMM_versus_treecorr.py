"""Compare TreeCorr to CLMM for stacked reduced tangential shear measurement.

This script compares, in a very ideal setup, the computation of the stacked
reduced shear profile either using CLMM or using TreeCorr to cross-correlate
cluster positions with background source shears. The mock data is generated
using CLMM.
"""
import os
import warnings

import healpy as hp
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
import scipy
import treecorr
from astropy.table import Table, vstack

import clmm
from clmm import ClusterEnsemble, Cosmology
from clmm.support import mock_data as mock
from clmm.utils.redshift_distributions import desc_srd

from jk_kmeans import jk_cov

PLOTS_DIR = "./plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# For reproducibility:
np.random.seed(11)

# -----------------------------------------------------------------------------
# Draw pairs of mass and redshift according to a mass function
# -----------------------------------------------------------------------------

# Mass function - CCL implementation
cosmo = ccl.Cosmology(
    Omega_c=0.265,
    Omega_b=0.0448,
    h=0.71,
    sigma8=0.8,
    n_s=0.96,
    Neff=3.04,
    m_nu=1.0e-05,
    mass_split="single",
)
hmd_200c = ccl.halos.MassDef(200, "critical")


# For a different multiplicty function, the user must change this function below
def tinker08_ccl(logm, z):
    mass = 10 ** (logm)
    hmf_200c = ccl.halos.MassFuncTinker08(mass_def=hmd_200c)
    nm = hmf_200c(cosmo, mass, 1.0 / (1 + z))
    return nm  # dn/dlog10M


# Computing the volume element
def dV_over_dOmega_dz(z):
    a = 1.0 / (1.0 + z)
    da = ccl.background.angular_diameter_distance(cosmo, a)
    E = ccl.background.h_over_h0(cosmo, a)
    return ((1.0 + z) ** 2) * (da**2) * ccl.physical_constants.CLIGHT_HMPC / cosmo["h"] / E


def pdf_tinker08_ccl(logm, z):
    return tinker08_ccl(logm, z) * dV_over_dOmega_dz(z)


# Acceptance-rejection method to sample (M,z) from the mass function
def bivariate_draw(
    pdf, N=1000, logm_min=14.0, logm_max=15.0, zmin=0.01, zmax=1, Ngrid_m=30, Ngrid_z=30
):
    """
    Uses the rejection method for generating random numbers derived from an arbitrary
    probability distribution.

    Parameters
    ----------
    pdf : func
      2d distribution function to sample
    N : int
      number of points to generate
    log_min,logm_max : float
      log10 mass range
    zmin,zmax : float
      redshift range

    Returns:
    --------
    ran_logm : list
        accepted logm values
    ran_z : list
        accepted redshift values
    acceptance : float
        acceptance ratio of the method
    """

    # maximum value of the pdf over the mass and redshift space.
    # the pdf is not monotonous in mass and redshift, so we need
    # to find the maximum numerically. Here we scan the space
    # with a regular grid and use the maximum value.
    # Accuracy of the results depends on Ngrid_m and Ngrid_z
    # This should probably be improved

    logM_arr = np.linspace(logm_min, logm_max, Ngrid_m)
    z_arr = np.logspace(np.log10(zmin), np.log10(zmax), Ngrid_z)
    p = []
    for logM in logM_arr:
        for z in z_arr:
            p.append(pdf(logM, z))
    pmax = np.max(p)
    pmin = np.min(p)
    # Counters
    naccept = 0
    ntrial = 0

    # Keeps drawing until N points are accepted
    ran_logm = []  # output list of random numbers
    ran_z = []  # output list of random numbers
    while naccept < N:
        # draw (logm,z) from uniform distribution
        # draw p from uniform distribution
        logm = np.random.uniform(logm_min, logm_max)  # x'
        z = np.random.uniform(zmin, zmax)  # x'
        p = np.random.uniform(0.0, pmax)  # y'

        if p < pdf(logm, z):
            # keep the point
            ran_logm.append(logm)
            ran_z.append(z)
            naccept = naccept + 1
        ntrial = ntrial + 1

    ran_logm = np.asarray(ran_logm)
    ran_z = np.asarray(ran_z)

    acceptance = float(N / ntrial)
    print(f"acceptance = {acceptance}")
    return ran_logm, ran_z, acceptance


# Draw N=30 (M,z) pairs

# Define the numbers of pairs to draw and the mass and redshift ranges
N = 100
logm_min = 14.5
logm_max = 14.8
zmin = 0.2
zmax = 0.3

pdf = pdf_tinker08_ccl

# Normalisation factor for the pdf to be used later
norm = (
    1.0
    / scipy.integrate.dblquad(
        pdf, zmin, zmax, lambda x: logm_min, lambda x: logm_max, epsrel=1.0e-4
    )[0]
)

# Random draw
ran_logm, ran_z, acceptance = bivariate_draw(
    pdf, N=N, logm_min=logm_min, logm_max=logm_max, zmin=zmin, zmax=zmax
)

# -----------------------------------------------------------------------------
# Generate a cluster catalog and associated source catalogs
# -----------------------------------------------------------------------------

# Cluster ensemble masses, redshifts and positions
cluster_m = 10 ** ran_logm
cluster_z = ran_z

# Concentration CCL object to compute the theoretical concentration
conc_obj = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200c)
conc_list = []
for number in range(0, len(cluster_m)):
    a = 1.0 / (1.0 + (cluster_z[number]))
    # mean value of the concentration for that cluster
    lnc_mean = np.log(conc_obj(cosmo, M=(cluster_m[number]), a=a))
    # random draw of actual concentration from normal distribution around lnc_mean, with a 0.14 scatter
    lnc = np.random.normal(lnc_mean, 0.14)
    conc_list.append(np.exp(lnc))

conc_list = np.array(conc_list)

# randomly draw cluster positions over the full sky
ra = np.random.random(N) * 360  # from 0 to 360 deg
sindec = np.random.random(N) * 2 - 1
dec = np.arcsin(sindec) * 180 / np.pi  # from -90 to 90 deg

print(len(cluster_m), N)

# Background galaxy catalog generation
#
# For each cluster of the ensemble, we use `mock_data` to generate a background
# galaxy catalog and store the results in a `GalaxyCluster` object. Note that:
# - The cluster density profiles follow the NFW parametrisation
# - The source redshifts follow the Chang et al. distribution and have associated pdfs
# - The shapes include shape noise and shape measurement errors
# - Background galaxy catalogs are independent, even if the clusters are close
#   (i.e., no common galaxy between two catalogs).
# - For each cluster we then compute
#     - the tangential and cross Delta_Sigma for each background galaxy
#     - the weights `w_ls` to be used to compute the corresponding radial profiles
#
# The cluster objects are then stored in `gclist`.

warnings.simplefilter(
    "ignore"
)  # just to prevent warning print out when looping over the cluster ensemble below

gclist = []
tables = []
# number of galaxies in each cluster field (alternatively, can use the galaxy density instead)
ngal_density = 10
cosmo_clmm = Cosmology(H0=71.0, Omega_dm0=0.265 - 0.0448, Omega_b0=0.0448, Omega_k0=0.0)
cosmo_clmm.set_be_cosmo(cosmo)
gal_args = dict(
    cosmo=cosmo_clmm,
    zsrc="chang13",
    delta_so=200,
    massdef="critical",
    halo_profile_model="nfw",
    zsrc_max=2.0,
    field_size=16.0,
    shapenoise=0.04,
    photoz_sigma_unscaled=0.02,
    ngal_density=ngal_density,
    mean_e_err=0.1,
)

for i in range(N):
    # generate background galaxy catalog for cluster i
    cl = clmm.GalaxyCluster(
        f"mock_cluster_{i:04}",
        ra[i],
        dec[i],
        cluster_z[i],
        galcat=mock.generate_galaxy_catalog(
            cluster_m[i],
            cluster_z[i],
            conc_list[i],
            cluster_ra=ra[i],
            cluster_dec=dec[i],
            zsrc_min=0.35,  # cluster zmax = 0.3 in this example
            **gal_args,
        ),
    )

    # compute DeltaSigma for each background galaxy
    cl.compute_tangential_and_cross_components(
        shape_component1="e1",
        shape_component2="e2",
        tan_component="g_t",
        cross_component="g_x",
        cosmo=cosmo_clmm,
        is_deltasigma=False,
        use_pdz=True,
    )

    # compute the weights to be used to bluid the DeltaSigma radial profiles
    cl.compute_galaxy_weights(
        use_pdz=True,
        use_shape_noise=True,
        shape_component1="e1",
        shape_component2="e2",
        use_shape_error=True,
        shape_component1_err="e_err",
        shape_component2_err="e_err",
        weight_name="w_ls",
        cosmo=cosmo_clmm,
        is_deltasigma=False,
        add=True,
    )

    # append the cluster in the list
    gclist.append(cl)
    tables.append(cl.galcat)

# Stack as plain astropy Tables: GCData's metadata guard for "coordinate_system"
# rejects vstack's metadata merge even when all tables share the same value.
galcat_tot = vstack([Table(t) for t in tables])

# -----------------------------------------------------------------------------
# Create ClusterEnsemble object and estimation of individual excess surface
# density profiles
# -----------------------------------------------------------------------------

bins = np.logspace(np.log10(0.5), np.log10(60), 12)
ensemble_id = 1
clusterensemble = ClusterEnsemble(ensemble_id)
for cluster in gclist:
    clusterensemble.make_individual_radial_profile(
        galaxycluster=cluster,
        tan_component_in="g_t",
        cross_component_in="g_x",
        tan_component_out="g_t",
        cross_component_out="g_x",
        weights_in="w_ls",
        weights_out="W_l",
        bins=bins,
        bin_units="arcmin",
        cosmo=cosmo_clmm,
    )

# Stacked profile of the cluster ensemble
clusterensemble.make_stacked_radial_profile(tan_component="g_t", cross_component="g_x", weights="W_l")

# Jackknife covariance of the stack between radial bins
jk_n_side = 16
clusterensemble.compute_jackknife_covariance(
    n_side=jk_n_side, tan_component="g_t", cross_component="g_x"
)
njk = 10
kmeans_gt_cov, kmeans_gx_cov, km_centers = jk_cov(clusterensemble, njk=njk)
# -----------------------------------------------------------------------------
# Visualizing the stacked profiles
#
# In the figure below, we plot:
# - the individual g_t profiles of the clusters (light blue)
# - the stacked signal (red symbols)
# -----------------------------------------------------------------------------

moo = clmm.Modeling(massdef="critical", delta_mdef=200, halo_profile_model="nfw")
moo.set_cosmo(cosmo_clmm)
# Average values of mass and concentration of the ensemble to be used below
# to overplot the model on the stacked profile
moo.set_concentration(conc_list.mean())
moo.set_mass(cluster_m.mean())

r_stack, gt_stack, gx_stack = (clusterensemble.stacked_data[c] for c in ("radius", "g_t", "g_x"))
plt.rcParams["axes.linewidth"] = 2
fig, axs = plt.subplots(1, 2, figsize=(17, 6))

err_gt = clusterensemble.cov["tan_jk"].diagonal() ** 0.5
err_gx = clusterensemble.cov["cross_jk"].diagonal() ** 0.5
km_err_gt = np.sqrt(np.diag(kmeans_gt_cov))
km_err_gx = np.sqrt(np.diag(kmeans_gx_cov))
axs[0].errorbar(
    r_stack,
    gt_stack,
    err_gt,
    markersize=5,
    c="r",
    fmt="o",
    capsize=10,
    elinewidth=1,
    zorder=1000,
    alpha=1,
    label="Stack",
)
axs[0].errorbar(
    r_stack,
    gt_stack,
    km_err_gt,
    markersize=5,
    c="k",
    fmt="o",
    capsize=10,
    elinewidth=1,
    zorder=1000,
    alpha=1,
    label="Stack",
)
axs[1].errorbar(
    r_stack,
    gx_stack,
    err_gx,
    markersize=5,
    c="r",
    fmt="o",
    capsize=10,
    elinewidth=1,
    zorder=1000,
    alpha=1,
    label="Stack",
)
axs[1].errorbar(
    r_stack,
    gx_stack,
    km_err_gx,
    markersize=5,
    c="k",
    fmt="o",
    capsize=10,
    elinewidth=1,
    zorder=1000,
    alpha=1,
    label="Stack",
)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_xscale("log")


for i in range(N):
    axs[0].plot(
        clusterensemble.data["radius"][i],
        clusterensemble.data["g_t"][i],
        color="cyan",
        label="Individual",
        alpha=1,
        linewidth=1,
    )
    axs[1].plot(
        clusterensemble.data["radius"][i],
        clusterensemble.data["g_x"][i],
        color="cyan",
        label="Individual",
        alpha=1,
        linewidth=1,
    )
    if i == 0:
        axs[0].legend(frameon=False, fontsize=15)
        axs[1].legend(frameon=False, fontsize=15)

axs[0].set_xlabel("separation [arcmin]", fontsize=20)
axs[1].set_xlabel("separation [arcmin]", fontsize=20)
axs[0].tick_params(axis="both", which="major", labelsize=18)
axs[1].tick_params(axis="both", which="major", labelsize=18)
axs[0].set_ylabel(r"$g_+$", fontsize=20)
axs[1].set_ylabel(r"$g_\times$", fontsize=20)
axs[0].set_title(r"Tangential", fontsize=20)
axs[1].set_title(r"Cross", fontsize=20)

for ax in axs:
    ax.minorticks_on()
    ax.grid(lw=0.5)
    ax.grid(which="minor", lw=0.1)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "individual_and_stacked_profiles.png"))
plt.close(fig)

# -----------------------------------------------------------------------------
# Saving/Loading ClusterEnsemble
#
# The `ClusterEnsemble` object also has an option for saving/loading:
# -----------------------------------------------------------------------------

clusterensemble.save("ce.pkl")

# -----------------------------------------------------------------------------
# Compute reduced shear profile with TreeCorr and compare to CLMM stacked
# profile
# -----------------------------------------------------------------------------
# Cluster catalog
cat_cluster = treecorr.Catalog(
    ra=clusterensemble["ra"],
    dec=clusterensemble["dec"],
    ra_units="deg",
    dec_units="deg",
    patch_centers=km_centers
)
# Source catalog
cat_source = treecorr.Catalog(
    ra=galcat_tot["ra"],
    dec=galcat_tot["dec"],
    g1=galcat_tot["e1"],
    g2=galcat_tot["e2"],
    ra_units="deg",
    dec_units="deg",
    w=galcat_tot["w_ls"],
    patch_centers=cat_cluster.patch_centers
)



ng = treecorr.NGCorrelation(nbins=11, min_sep=0.5, max_sep=60, sep_units="arcmin", bin_type="Log")
ng.process(cat_cluster, cat_source)
theta = ng.meanr  # average angular separation in each bin
cov_jk = ng.estimate_cov('jackknife')
var_jk = np.diag(cov_jk)  
print(cosmo_clmm.be_cosmo)

da = cosmo.angular_diameter_distance(1.0 / (1.0 + np.mean(clusterensemble["z"])))
da_clmm = cosmo_clmm.eval_da(np.mean(clusterensemble["z"]))
print(da, da_clmm)

arcmin_to_Mpc = np.pi / (60.0 * 180) * da_clmm

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.errorbar(
    theta,
    ng.xi,
    np.sqrt(var_jk),
    color="blue",
    marker="x",
    markersize=15,
    linestyle="",
    label="Treecorr - cluster positions x source shears",
)

ax.errorbar(
    r_stack,
    gt_stack,
    err_gt,
    markersize=5,
    c="r",
    fmt=".",
    capsize=10,
    elinewidth=1,
    zorder=1000,
    alpha=1,
    label="CLMM - stacked reduced rangential shear",
)

ax.plot(
    clusterensemble.data["radius"][0],
    moo.eval_reduced_tangential_shear(
        clusterensemble.data["radius"][0] * arcmin_to_Mpc,
        cluster_z.mean(),
        z_src=desc_srd,
        z_src_info="distribution",
    ),
    "--k",
    linewidth=1,
    label="CLMM prediction for stack 'mean' cluster",
    zorder=100,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([-0.02, 0.15])
ax.axhline(0, color="k", ls=":")
ax.set_xlabel("separation [arcmin]")
ax.set_ylabel("$\\widehat{g_t}$")
ax.legend()
fig.savefig(os.path.join(PLOTS_DIR, "treecorr_vs_clmm.png"))
plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(err_gt, np.sqrt(var_jk), label='healpy')
ax.scatter(km_err_gt, np.sqrt(var_jk), label='kmeans')
ax.plot([0, 0.007], [0, 0.007], "k:")
ax.set_xlabel("Error from CLMM")
ax.set_ylabel("Error from TreeCorr")
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
fig.savefig(os.path.join(PLOTS_DIR, "error_comparison.png"))
plt.close(fig)
