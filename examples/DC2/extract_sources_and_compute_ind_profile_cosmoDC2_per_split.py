# -*- coding: utf-8 -*-
import numpy as np
import GCRCatalogs
#GCRCatalogs.set_root_dir_by_site('in2p3')
import healpy
import glob, sys
import astropy.units as u
from astropy.io import fits as fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy
import clmm
import pandas as pd

from astropy.table import QTable, Table, vstack, join, hstack
import pickle, sys

def load(filename, **kwargs):
    """Loads GalaxyCluster object to filename using Pickle"""
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

from clmm import Cosmology
from scipy.integrate import simps
#cosmoDC2 cosmology
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
#connection with qserv
import mysql
from mysql.connector import Error
import argparse
import _utils_cosmoDC2

mag_i_max =  24.25
mag_r_max =  28
def source_selection_magnitude(table):
    mask = table['mag_i'] < mag_i_max
    mask *= table['mag_r'] < mag_r_max
    return table[mask]

def collect_argparser():
    parser = argparse.ArgumentParser(description="Let's extract source catalogs and estimate individual lensing profiles")
    parser.add_argument("--which_split", type=int, required=True)
    parser.add_argument("--number_of_splits", type=int, required=True)
    parser.add_argument("--lens_catalog_name", type=str, required=False, default='./data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl',)
    parser.add_argument("--lambda_min", type=float, required=False, default=20,)
    parser.add_argument("--lambda_max", type=float, required=False, default=40,)
    parser.add_argument("--redshift_min", type=float, required=False, default=0.2,)
    parser.add_argument("--redshift_max", type=float, required=False, default=0.4,)
    return parser.parse_args()

#select galaxy clusters
_config_extract_sources_in_cosmoDC2 = collect_argparser()
lens_catalog_name=_config_extract_sources_in_cosmoDC2.lens_catalog_name
lens_catalog=np.load(lens_catalog_name, allow_pickle=True)
mask_select=(lens_catalog['richness'] > _config_extract_sources_in_cosmoDC2.lambda_min)
mask_select*=(lens_catalog['richness'] < _config_extract_sources_in_cosmoDC2.lambda_max)
mask_select*=(lens_catalog['redshift'] > _config_extract_sources_in_cosmoDC2.redshift_min)
mask_select*=(lens_catalog['redshift'] < _config_extract_sources_in_cosmoDC2.redshift_max)
lens_catalog=lens_catalog[mask_select]
lens_catalog_truncated=lens_catalog

n_cl=len(lens_catalog_truncated)
index_cl=np.arange(n_cl)
split_lists=np.array_split(index_cl, int(_config_extract_sources_in_cosmoDC2.number_of_splits))
lens_catalog_truncated=lens_catalog_truncated[np.array(split_lists[int(_config_extract_sources_in_cosmoDC2.which_split)])]
n_cl_to_extract=len(lens_catalog_truncated)

path_where_to_save_profiles = f'./data/individual_lensing_profiles/'
name_save = path_where_to_save_profiles+f'individual_profiles'
name_save +=f'_split={_config_extract_sources_in_cosmoDC2.which_split}'
name_save +=f'_number_of_splits={_config_extract_sources_in_cosmoDC2.number_of_splits}_ncl={n_cl_to_extract}.pkl'

print(f'[cluster catalog]: {_config_extract_sources_in_cosmoDC2.lambda_min} < richness < {_config_extract_sources_in_cosmoDC2.lambda_max}')
print(f'[cluster catalog]: {_config_extract_sources_in_cosmoDC2.redshift_min} < redshift < {_config_extract_sources_in_cosmoDC2.redshift_max}')
print(f'[cluster catalog]: number of clusters = {n_cl}')
print(f'[cluster catalog]: number of clusters in split {_config_extract_sources_in_cosmoDC2.which_split} = {n_cl_to_extract}')
print(f'[individual lensing profiles]: individual profiles ' + name_save)

for n, lens in enumerate(lens_catalog_truncated):

    z, ra, dec=lens['redshift'], lens['ra'], lens['dec']
    cluster_id=lens['cluster_id']
    richness=lens['richness']
    print(f'[load background source catalog]: for redMaPPer cluster {cluster_id}')
    #cluster member galaxies
    id_member_galaxy = lens_catalog['id_member'][n]
    ra_member_galaxy = lens_catalog['ra_member'][n]
    dec_member_galaxy = lens_catalog['dec_member'][n]
    lens_distance=cosmo.eval_da(z)
    
    tab = _utils_cosmoDC2.extract_cosmoDC2_galaxy(lens_z, lens_distance, ra, dec, rmax = 10, method = 'qserv')
    #compute reduced shear and ellipticities
    tab['g1'], tab['g2'] = clmm.utils.convert_shapes_to_epsilon(tab['shear1'],tab['shear2'], 
                                                                shape_definition = 'shear',
                                                                kappa = tab['kappa'])
    ellipticity_uncorr_e1 = tab['e1_true_uncorr']
    ellipticity_uncorr_e2 = tab['e2_true_uncorr']
    ellipticity_corr_e1, ellipticity_corr_e2 = _utils_cosmoDC2.correct_shear_ellipticity(ellipticity_uncorr_e1, ellipticity_uncorr_e2)
    tab['e1_true'] = ellipticity_corr_e1
    tab['e2_true'] = ellipticity_corr_e2
    tab['e1'], tab['e2'] = clmm.utils.compute_lensed_ellipticity(tab['e1_true'], tab['e2_true'], 
                                                                 tab['shear1'], tab['shear2'], 
                                                                 tab['kappa'])

    dat_extract_mag_cut = source_selection_magnitude(tab)
    mask_z = dat_extract_mag_cut['z'] > z + 0.2
    dat_extract_mag_cut_z_cut = dat_extract_mag_cut[mask_z]
    
    dat = clmm.GCData(
    [
        dat_extract_mag_cut_z_cut["ra"],
        dat_extract_mag_cut_z_cut["dec"],
        dat_extract_mag_cut_z_cut["e1"],
        dat_extract_mag_cut_z_cut["e2"],
        dat_extract_mag_cut_z_cut["z"],
        dat_extract_mag_cut_z_cut["galaxy_id"],
    ],
    names=("ra", "dec", "e1", "e2", "z", "id"),
    masked=True,)

    cl = clmm.GalaxyCluster("redmapper_cluster", ra, dec, z, dat)
    
    bin_edges = clmm.dataops.make_bins(0.5, 30, 15, method='evenlog10width')
    
    cl.compute_tangential_and_cross_components(
                                                shape_component1="e1",
                                                shape_component2="e2",
                                                tan_component="DS_t",
                                                cross_component="DS_x",
                                                cosmo=cosmo,
                                                is_deltasigma=True,
                                                use_pdz=False,)
    
    cl.compute_galaxy_weights(
                                use_pdz=False,
                                use_shape_noise=False,
                                shape_component1="e1",
                                shape_component2="e2",
                                use_shape_error=False,
                                shape_component1_err="e_err",
                                shape_component2_err="e_err",
                                weight_name="w_ls",
                                cosmo=cosmo,
                                is_deltasigma=True,
                                add=True,)
    
    cl.make_radial_profile("Mpc",
                            bins=bin_edges,
                            error_model="ste",
                            cosmo=cosmo,
                            tan_component_in="DS_t",
                            cross_component_in="DS_x",
                            tan_component_out="binned_DS_t",
                            cross_component_out="binned_DS_x",
                            tan_component_in_err=None,
                            cross_component_in_err=None,
                            include_empty_bins=False,
                            gal_ids_in_bins=False,
                            add=True,
                            table_name="profile",
                            overwrite=True,
                            use_weights=True,
                            weights_in="w_ls",
                            weights_out="W_l",)
    
    binned_DS_t = cl.profile['binned_DS_t']
    binned_DS_x = cl.profile['binned_DS_x']
    W_l = cl.profile['W_l']
    radius = cl.profile['radius']
    cluster_data = [cluster_id, ra, dec, z, lens['richness']]
    cluster_data_colnames = ['cluster_id', 'cluster_ra', 'cluster_dec', 'cluster_redshift', 'cluster_richness']
    if n==0:
        ind_profile = {k:[] for k in list(cl.profile.colnames)+cluster_data_colnames}
    
    for index, name in enumerate(cl.profile.colnames): 
        ind_profile[name].append(cl.profile[name])
    for index, name in enumerate(cluster_data_colnames): 
        ind_profile[name].append(cluster_data[index])

save_pickle(Table(ind_profile), name_save)

            

    

    

    

    

    

    

    

    
