import GCRCatalogs
GCRCatalogs.set_root_dir_by_site('in2p3')
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table
def load(filename, **kwargs):
    """Loads GalaxyCluster object to filename using Pickle"""
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)
def save_pickle(dat, filename, **kwargs):
     file = open(filename,'wb')
     pickle.dump(dat, file)
     file.close()
catalog = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_redmapper_v0.8.1')
quantity = ['cluster_id','ra', 'dec',
            'redshift', 'redshift_err', 
            'richness', 'richness_err',
            'ra_member', 'dec_member',
            'id_member','cluster_id_member']
dat = catalog.get_quantities(quantity)

line_ra_member = []
line_dec_member = []
line_id_member = []
line_cluster_id_member = []
for i in range(len(dat['cluster_id'])):
    mask  = (dat['cluster_id_member']==dat['cluster_id'][i])
    line_ra_member.append(dat['ra_member'][mask]) 
    line_dec_member.append(dat['dec_member'][mask]) 
    line_id_member.append(dat['id_member'][mask])
    line_cluster_id_member.append(dat['cluster_id_member'][mask])

dictionnary = {'cluster_id': dat['cluster_id'],
               'ra': dat['ra'], 
               'dec': dat['dec'], 
               'redshift': dat['redshift'],
               'redshift_err': dat['redshift_err'], 
               'richness': dat['richness'], 
               'richness_err': dat['richness_err'],
               'ra_member': line_ra_member,
               'dec_member': line_dec_member,
               'id_member': line_id_member,
               'cluster_id_member': line_cluster_id_member,}

t = Table(dictionnary)

save_pickle(t, './DC2_data/cosmoDC2_v1.1.4_redmapper_v0.8.1_catalog.pkl')
