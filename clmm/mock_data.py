import sys
from clmm.models import CLMM_densityModels_beforeConvertFromPerH as clmm_mod
import numpy as np
import colossus.cosmology.cosmology as Cosmology
import matplotlib.pyplot as plt
from astropy.table import Table


class MockData(): 
    """
    a class that generates a mock background galaxy catalog around a galaxy cluster
    
    Attributes
    ----------
    config: dictionary
        The configuration to generate the fake data. The cluster is located in (0,0).
        The fields of the dictionary should be:
        ngals: int
            number of galaxies in the fake catalog
        cluster_m: float
            mass of the cluster
        cluster_z: float
            redshift of cluster
        concentration: float
            concentration of the cluster
        src_z: float
            redshift of background galaxies
        cosmo: string 
            Defines the cosmological parameter set in colossus, e.g. WMAP7-ML
        mdef: string
            Mass definition, e.g. '200c'
            
    catalog: astropy table
        The catalog generated given the user-defined configuration

    """

    
    def __init__(self, config=None):
        """
        Parameters
        ----------

        """
        if config is not None:
            self.config = config
        else:
            self.config = {}
            self.config['ngals'] = int(3.e4)
            self.config['cluster_m'] = 1.e15
            self.config['cluster_z'] = 0.3
            self.config['src_z'] = 0.8
            self.config['cosmo'] = 'WMAP7-ML'
            self.config['mdef'] = '200c'
            self.config['concentration'] = 4
            

    def generate(self, is_shapenoise=False, shapenoise=0.005, is_zerr=False, sigma_z=0.05):
        """
        This generates a fake dataset of background galaxies using the Dallas group software. 
        By default, data are perfect, i.e. no shape noise, all galaxies at the same redshift.
        TODO: implement the shape noise, and redshift error options

        Options:
        --------
        is_shape_noise: Boolean
            If True, noise is added to the galaxy shapes
        is_z_err: Boolean
            If True, a photometric redshift error is added to the background galaxy redshifts
        
        Returns:
        ---------
            astropy table containing the galaxy catalog [id, ra, dec, gamma1, gamma2, z]
        """

        ngals = self.config['ngals']
   
        self.z_best = np.zeros(ngals)+self.config['src_z']
    
        if is_zerr:
            # introduce a redshift error on the source redshifts and generate 
            # the corresponding Gaussian pdf
            self.config['redshift_err'] = sigma_z
            self.z_true = self.config['src_z'] + sigma_z*np.random.standard_normal(ngals)
            zmin = self.config['src_z'] - 0.3
            zmax = self.config['src_z'] + 0.3
            self.zbins = np.arange(zmin, zmax, 0.02)
            self.z_pdf = np.exp(-0.5*((self.zbins - self.config['src_z'])/sigma_z)**2)/np.sqrt(2*np.pi*sigma_z**2)    
            assert np.abs(np.trapz(self.z_pdf, self.zbins) - 1) < 1e-6
            self.pdf_grid = np.vstack(ngals*[self.z_pdf])
        else:
            # No redshift error
            self.z_true = self.z_best

        seqnr = np.arange(ngals)
        zL = self.config['cluster_z'] # cluster redshift
        mdef = self.config['mdef']
        cosmo = Cosmology.setCosmology(self.config['cosmo'])

        M = self.config['cluster_m']*cosmo.h
        c = self.config['concentration']  
        r = np.linspace(0.25, 10., 1000) #Mpc
        r = r*cosmo.h #Mpc/h

        testProf = clmm_mod.nfwProfile(M = M, c = c, zL = zL, mdef = mdef, \
                                chooseCosmology = self.config['cosmo'], esp = None)

        x_mpc = np.random.uniform(-4, 4, size=ngals)
        y_mpc = np.random.uniform(-4, 4, size=ngals)
        r_mpc = np.sqrt(x_mpc**2 + y_mpc**2)

        Dl = cosmo.angularDiameterDistance(zL)

        x_deg = (x_mpc/Dl)*(180./np.pi) #ra
        y_deg = (y_mpc/Dl)*(180./np.pi) #dec

        gamt= testProf.deltaSigma(r_mpc)/testProf.Sc(self.z_true)
        if is_shapenoise:
            self.config['shape_noise'] = shapenoise
            gamt = gamt + shapenoise*np.random.standard_normal(ngals)

        posangle = np.arctan2(y_mpc, x_mpc)
        cos2phi = np.cos(2*posangle)
        sin2phi = np.sin(2*posangle)

        e1 = -gamt*cos2phi
        e2 = -gamt*sin2phi

        if is_zerr:
            self.catalog = Table([seqnr, -x_deg, y_deg, e1, e2, self.z_best, self.pdf_grid], \
                                  names=('id', 'ra','dec','gamma1','gamma2', 'z', 'z_pdf'))
        else:
            self.catalog = Table([seqnr, -x_deg, y_deg, e1, e2, self.z_best], \
                                names=('id', 'ra','dec','gamma1','gamma2', 'z'))
