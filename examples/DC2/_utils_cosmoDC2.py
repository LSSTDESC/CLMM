import sys, os
import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import pandas as pd
import clmm
import cmath
import GCRCatalogs
r"""
extract background galaxy catalog with qserv for:
cosmodc2:
- true shapes
- true redshift
and GCRCatalogs:
- photoz addons
"""
def _fix_axis_ratio(q_bad):
    # back out incorrect computation of q using Johnsonb function
    e_jb = np.sqrt((1 - q_bad**2)/(1 + q_bad**2))
    q_new = np.sqrt((1 - e_jb)/(1 + e_jb)) # use correct relationship to compute q from e_jb 
    return q_new

def _fix_ellipticity_disk_or_bulge(ellipticity):
    # back out incorrect computation of q using Johnsonb function 
    q_bad = (1-ellipticity)/(1+ellipticity) #use default e definition to calculate q
    # q_bad incorrectly computed from e_jb using q_bad = sqrt((1 - e_jb^2)/(1 + e_jb^2))
    q_new = _fix_axis_ratio(q_bad)
    e_new = (1 - q_new)/(1 + q_new)  # recompute e using default (1-q)/(1+q) definition
    return e_new

def correct_shear_ellipticity(ellipticity_uncorr_e1, ellipticity_uncorr_e2):
    ellipticity_uncorr_norm = (ellipticity_uncorr_e1**2+ellipticity_uncorr_e2**2)**.5
    complex_ellipticity_uncorr = ellipticity_uncorr_e1 + 1j*ellipticity_uncorr_e2
    phi = np.array([cmath.phase(c) for c in complex_ellipticity_uncorr])
    ellipticity_corr_norm = _fix_ellipticity_disk_or_bulge(ellipticity_uncorr_norm)
    ellipticity_corr = ellipticity_corr_norm*np.exp(1j*phi)
    ellipticity_corr_e1, ellipticity_corr_e2 = ellipticity_corr.real, ellipticity_corr.imag
    return ellipticity_corr_e1, ellipticity_corr_e2

def extract_photoz(id_gal, healpix=None, GCRcatalog=None):
    r"""
    extract background galaxy catalog with GCRcatalog (healpix subdivision)
    Attributes:
    -----------
    id_gal: array
        background galaxy id
    healpix: array
        list of healpix pixels where to find galaxies
    GCRcatalog: GCRcatalog
        background galaxy GCRcatalog object
    Returns:
    --------
    tab_astropy_ordered: Table
        photoz informations 
    """
    Table_id_gal = Table()
    Table_id_gal['galaxy_id'] = id_gal
    quantities_photoz=['photoz_pdf','photoz_mean','photoz_mode',
                       'photoz_odds','galaxy_id']
    z_bins = GCRcatalog.photoz_pdf_bin_centers
    z_bins[0] = 1e-7
    for n, hp in enumerate(np.array(healpix)):
        tab = GCRcatalog.get_quantities(quantities_photoz, 
                                        native_filters=['healpix_pixel=='+str(hp)])
        tab_astropy = Table()
        tab_astropy['galaxy_id']   = tab['galaxy_id']
        tab_astropy['photoz_pdf']  = tab['photoz_pdf']
        tab_astropy['photoz_mean'] = tab['photoz_mean']
        tab_astropy['photoz_mode'] = tab['photoz_mode']
        tab_astropy['photoz_odds'] = tab['photoz_odds']
        mask_id=np.isin(np.array(tab_astropy['galaxy_id']), id_gal)
        tab_astropy=tab_astropy[mask_id]
        if n==0: 
            table_photoz=tab_astropy
        else: 
            tab_=vstack([table_photoz,tab_astropy])
            table_photoz = tab_
            
    n_gal = len(table_photoz['galaxy_id'])
    table_photoz['pzbins'] = np.array([z_bins for i in range(n_gal)])
    table_photoz_ordered = join(table_photoz, Table_id_gal, keys='galaxy_id')
    return table_photoz_ordered

def extract(lens_redshift=None,
            qserv_query=None, GCRcatalog=None, conn_qserv=None,
           cosmo=None):
    r"""
    extract background galaxy catalog
    Attributes:
    -----------
    lens_redshift: float
        lens redshift
    lens_ra: float
        lens right ascension
    lens_dec: float
        lens declinaison
    Returns:
    --------
    cl: GalaxyCluster
        background galaxy catalog
    """
    #qserv
    query_mysql = qserv_query
    tab = pd.read_sql_query(query_mysql, conn_qserv)
    try: 
        tab = QTable.from_pandas(tab)
    except: 
        print('no data')
        return None
    #compute reduced shear and ellipticities
    tab['g1'], tab['g2'] = clmm.utils.convert_shapes_to_epsilon(tab['shear1'],tab['shear2'], 
                                                                shape_definition = 'shear',
                                                                kappa = tab['kappa'])
    ellipticity_uncorr_e1 = tab['e1_true_uncorr']
    ellipticity_uncorr_e2 = tab['e2_true_uncorr']
    ellipticity_corr_e1, ellipticity_corr_e2 = correct_shear_ellipticity(ellipticity_uncorr_e1, ellipticity_uncorr_e2)
    tab['e1_true'] = ellipticity_corr_e1
    tab['e2_true'] = ellipticity_corr_e2
    tab['e1'], tab['e2'] = clmm.utils.compute_lensed_ellipticity(tab['e1_true'], tab['e2_true'], 
                                                                 tab['shear1'], tab['shear2'], 
                                                                 tab['kappa'])
    return tab

def extract_photoz(z_cl, pz_catalog=None, z_bins_pz=None, healpix_list=None, 
                   id_gal_to_extract=None, _query_photoz=None, cosmo=None):
    pz_table = Table(names = ['sigmac_photoz', 'p_background', 'photoz_dispersion', 
                                  'sigmac_estimate_0', 'sigmac_estimate_1', 'sigmac_estimate_2', 
                                  'z_estimate_0', 'z_estimate_1', 'z_estimate_2', 
                                  'galaxy_id', 'photoz_mean', 'photoz_mode', 'photoz_odds'])
    photoz_gc_ = GCRCatalogs.load_catalog(pz_catalog)
    for i, hp in enumerate(healpix_list):
        #browse healpix pixels
        print(f'-----> heapix pixel = ' + str(hp))
        chunk = photoz_gc_.get_quantities(_query_photoz, native_filters=[f'healpix_pixel=={hp}'], return_iterator=True)
        for j in range(3):
            #browse chunk data
            print('chunk = ' + str(j))
            try: 
                dat_extract_photoz_chunk = Table(next(chunk))
            except: continue
            print(f'number of galaxies in the full healpix = ' + str(len(dat_extract_photoz_chunk['galaxy_id'])))
            #use only selected galaxies
            dat_extract_photoz_chunk_truncated = dat_extract_photoz_chunk[np.isin(dat_extract_photoz_chunk['galaxy_id'], id_gal_to_extract)]
            if len(dat_extract_photoz_chunk_truncated['galaxy_id']) == 0: continue
            print('number of galaxies tageted in the healpix = ' + str(len(dat_extract_photoz_chunk_truncated['galaxy_id'])))
            pzbins_table=np.array([z_bins_pz for i in range(len(dat_extract_photoz_chunk_truncated['photoz_pdf'].data))])
            #compute WL weights with 
            pz_quantities_chunk = _utils_photometric_redshifts.compute_photoz_quantities(z_cl, dat_extract_photoz_chunk_truncated['photoz_pdf'], 
                                                                   pzbins_table, n_samples_per_pdf=3, cosmo=cosmo,
                                                                   use_clmm=False)
            pz_quantities_chunk['galaxy_id'] = dat_extract_photoz_chunk_truncated['galaxy_id']
            pz_quantities_chunk['photoz_mean'] = dat_extract_photoz_chunk_truncated['photoz_mean']
            pz_quantities_chunk['photoz_mode'] = dat_extract_photoz_chunk_truncated['photoz_mode']
            pz_quantities_chunk['photoz_odds'] = dat_extract_photoz_chunk_truncated['photoz_odds']
            pz_table = vstack([pz_table, pz_quantities_chunk])
    return pz_table
