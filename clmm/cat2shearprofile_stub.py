
def read_catalog():
    """ 
    This reads in the galaxy catalog, which can either:
    - DC2 extragalactic catalog (shear1, shear2)
    - DC2 coadd catalog (shapeHSM_e1,shapeHSM_e2)
    - DM stack catalog (shapeHSM_e1,shapeHSM_e2)
    - ...
    
    Returns:
    - Homogenised astropy table containing at least: ra[deg], dec[deg], e1, e2, (z or p(z)), relevant flags...
 
    """
    
def compute_shear():
    """
    Input: galaxy catalog astropy table
    Output: new columns on the astropy table containing 
        - tangential shear 
        - cross shear
        - physical projected distance to cluster center
        
    phi = numpy.arctan2(dec, ra)
    gamt = - (e1 * numpy.cos(2.0 * phi) + e2 * numpy.sin(2.0 * phi))
    gamc =  e1 * numpy.sin(2.0 * phi) - e2 * numpy.cos(2.0 * phi)
    dist = fcn(cosmology, phi, z)
    """
    
def make_shear_profile():
    """
    Input: 
    - galaxy catalog astropy table, including results from compute_shear()
    - is_binned flag
    
    Options:
    - bin edge array
    
    If is_binned == False: simply returns dist and gamt from astropy table
    If is_binned == True: average gamt in each bin defined by bin edge array
    """

def make_bins():
    """
    Bin definition. Various options:
    - User-defined number of galaxies in each bin
    - regular radial binning
    - log binning
    - anything else?
    
    Returns array of bin edges (physical units), to be used in make_shear_profile 
    """
