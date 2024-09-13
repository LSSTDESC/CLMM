import numpy as np
from astropy.table import QTable, Table, vstack, join
import pickle 
import pandas as pd
import clmm
import cmath
import GCRCatalogs
GCRCatalogs.set_root_dir_by_site('in2p3')
import mysql
from mysql.connector import Error

def _fix_axis_ratio(q_bad):
    
    #from https://github.com/LSSTDESC/gcr-catalogs/blob/ellipticity_bug_fix/GCRCatalogs/cosmodc2.py 
    # back out incorrect computation of q using Johnsonb function
    e_jb = np.sqrt((1 - q_bad**2)/(1 + q_bad**2))
    q_new = np.sqrt((1 - e_jb)/(1 + e_jb)) # use correct relationship to compute q from e_jb 
    return q_new

def _fix_ellipticity_disk_or_bulge(ellipticity):
    
    #from https://github.com/LSSTDESC/gcr-catalogs/blob/ellipticity_bug_fix/GCRCatalogs/cosmodc2.py 
    # back out incorrect computation of q using Johnsonb function 
    q_bad = (1-ellipticity)/(1+ellipticity) #use default e definition to calculate q
    # q_bad incorrectly computed from e_jb using q_bad = sqrt((1 - e_jb^2)/(1 + e_jb^2))
    q_new = _fix_axis_ratio(q_bad)
    e_new = (1 - q_new)/(1 + q_new)  # recompute e using default (1-q)/(1+q) definition
    return e_new

def correct_shear_ellipticity(ellipticity_uncorr_e1, ellipticity_uncorr_e2):
    
    #from https://github.com/LSSTDESC/gcr-catalogs/blob/ellipticity_bug_fix/GCRCatalogs/cosmodc2.py 
    ellipticity_uncorr_norm = (ellipticity_uncorr_e1**2+ellipticity_uncorr_e2**2)**.5
    complex_ellipticity_uncorr = ellipticity_uncorr_e1 + 1j*ellipticity_uncorr_e2
    phi = np.array([cmath.phase(c) for c in complex_ellipticity_uncorr])
    ellipticity_corr_norm = _fix_ellipticity_disk_or_bulge(ellipticity_uncorr_norm)
    ellipticity_corr = ellipticity_corr_norm*np.exp(1j*phi)
    ellipticity_corr_e1, ellipticity_corr_e2 = ellipticity_corr.real, ellipticity_corr.imag
    return ellipticity_corr_e1, ellipticity_corr_e2

def extract_cosmoDC2_galaxy(lens_z, lens_distance, ra, dec, rmax = 10, method = 'qserv'):
    
    if method == 'qserv':
        
        def qserv_query(lens_z, lens_distance, ra, dec, rmax = 10):
            r"""
            quantities wanted + cuts for qserv
            Attributes:
            -----------
            z: float
                lens redshift
            lens_distance: float
                distance to the cluster
            ra: float
                lens right ascension
            dec: float
                lens declinaison
            rmax: float
                maximum radius
            """
            zmax = 3.
            zmin = lens_z
            theta_max = (rmax/lens_distance) * (180./np.pi)
            query = "SELECT data.coord_ra as ra, data.coord_dec as dec, data.redshift as z, "
            query += "data.galaxy_id as galaxy_id, data.halo_id as halo_id, "
            query += "data.mag_i, data.mag_r, data.mag_y, "
            query += "data.shear_1 as shear1, data.shear_2 as shear2, data.convergence as kappa, "
            query += "data.ellipticity_1_true as e1_true_uncorr, data.ellipticity_2_true as e2_true_uncorr " 
            query += "FROM cosmoDC2_v1_1_4_image.data as data "
            query += f"WHERE data.redshift >= {zmin} AND data.redshift < {zmax} "
            query += f"AND scisql_s2PtInCircle(coord_ra, coord_dec, {ra}, {dec}, {theta_max}) = 1 "
            query += f"AND data.mag_i <= 24.6 "
            query += f"AND data.mag_r <= 28.0 "
            query += ";" 
            return query
        
        conn_qserv = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
        cursor = conn_qserv.cursor(dictionary=True, buffered=True)
        qserv_query_extract = qserv_query(z, lens_distance, ra, dec, rmax=30)
        tab = pd.read_sql_query(qserv_query_extract, conn_qserv)
        tab = QTable.from_pandas(tab)
        return tab
    
    
     if method == 'gcrcatalogs':
        
        raise ValueError("Extraction with GCRCatalogs not implemented yet")
        
#         return 
        
#         def query_photoz():
    
#             return ['photoz_pdf', 'photoz_mean','photoz_mode','photoz_odds','galaxy_id']
        
#         #generate random
#         n_random = 
#         ras, decs
#         healpix_dc2 = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image").get_catalog_info()['healpix_pixels']
#         cosmoDC2 = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image")
#         healpix = np.unique(healpy.ang2pix(32, ras, decs, nest=False, lonlat=True))
#         healpix_list = healpix[np.isin(healpix, healpix_dc2)]
#         ra_min, ra_max = ra_cl - 0.3, ra_cl + 0.3
#         dec_min, dec_max = dec_cl - 0.3, dec_cl + 0.3
#         z_min = z_cl + 0.1
#         mag_i_max = 25

#         coord_filters = [
#             "ra >= {}".format(ra_min),
#             "ra < {}".format(ra_max),
#             "dec >= {}".format(dec_min),
#             "dec < {}".format(dec_max),]
        
#         z_filters = ["redshift >= {}".format(z_min)]
#         mag_filters = ["mag_i < {}".format(mag_i_max)]
        
#         for i, hp in enumerate(healpix_list):
#             print(f'-----> heapix pixel = ' + str(hp))
#             chunk = cosmoDC2.get_quantities(_query_photoz, filters=(coord_filters + z_filters + mag_filters), 
#                                             native_filters=[f'healpix_pixel=={hp}'], return_iterator=True)
#             for j in range(3):
        
        
    