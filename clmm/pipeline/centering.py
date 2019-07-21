import os
import numpy as np
from astropy.table import Table
from .cluster import Cluster

def mean_centering(file_dir, **save_kwargs):
    cluster_list = [f for f in os.listdir(file_dir) if f[-5:]=='.fits']
    cluster_centers = np.zeros((len(cluster_list),2))
    
    cl = Cluster()
    for i in range(len(cluster_list)):
        cl.load_gals(os.path.join(file_dir, cluster_list[i]))
        
        cluster_centers[i,0] = np.mean(cl.gals['ra'])
        cluster_centers[i,1] = np.mean(cl.gals['dec'])
    
    table = Table([cluster_centers[:,0], cluster_centers[:,1]], names=('ra', 'dec'), meta={'name': 'cluster center'})
    table['ra'].unit = 'deg'
    table['dec'].unit = 'deg'
    
    table.write('cluster_centers.fits', **save_kwargs)