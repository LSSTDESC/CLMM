import sys
from clmm.models import CLMM_densityModels_beforeConvertFromPerH as clmm_mod
from clmm.core import CLMMBase
import numpy as np
import colossus.cosmology.cosmology as Cosmology
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy import integrate
from scipy.interpolate import interp1d


class MockData(CLMMBase): 
    '''
    A class that generates a mock background galaxy catalog around a galaxy cluster
    
    Attributes
    ----------
    config: dictionary
        Main properties of the mock data setup (number of galaxies, cluster mass,
        cluster redshift, cluster concentration, source redshift, cosmo. params., 
        mass definition)        
    catalog: astropy table
        The catalog generated given the user-defined configuration
    '''

    
    def __init__(self, config=None):
        '''
        Parameters
        ----------
        config: dictionary
            Main properties of the mock data setup. The cluster is located in (0,0).
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
        '''
        
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

        self.ask_type = ['raw_data']

    def generate(self, is_shapenoise=False, shapenoise=0.005, is_zerr=False, sigma_z_ref=0.05, 
                 is_zdistribution=False, z_min=0., z_max=7., alpha=1.24, beta=1.01, z0=0.51):
        '''
        Generates a mock dataset of sheared background galaxies using the Dallas group software. 

        Parameters
        ----------
        is_shapenoise: bool, optional
            If True, noise is added to the galaxy shapes
        shapenoise: float, optional
            Amount of noise to add to the galaxy shapes
        is_zerr: bool, optional
            If True, a photometric redshift error is added to the background galaxy redshifts
            and a gaussian pdf is created
        sigma_z_ref: float, optional
            Width of the redshift Gaussian pdf, o be scaled by (1+z)
        is_zdistribution: bool, optional
            Default is for single background sources redshift. 
            If True, the redshifts of background sources is taken from
            a redshift distribution (Default values taken from Chang et al. (2013) eq. 21).
        z_min: float, optional
            = 0.
        z_max: float, optional
            = 7. 
        
        alpha: float, optional
            Default value taken from Chang et al. (2013) eq. 21
        beta: float, optional
            Default value taken from Chang et al. (2013) eq. 21
        z0: float, optional
            Default value taken from Chang et al. (2013) eq. 21
        '''

        
        ngals = self.config['ngals']
   
        
        def pzfxn(z):
            #Form of redshift distribution function 
            return (z**alpha)*np.exp(-(z/z0)**beta)
            
        def integrated_pzfxn(zmax,fxn):
            #Integrated redshift distribution function for transformation method
            I, err = integrate.quad(fxn,self.config['cluster_z']+0.1,zmax)
            return I
        
        if is_zdistribution is True:
            # https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf
            # Numerical Recipe in C, Chap 7.2: Transformation Method. Pulling out random values from a given probability distribution
            
            # sampling the domain of the original p(z)
            z_min = self.config['cluster_z'] + 0.1 # fix z_min to get background galaxies only
            z_domain = np.arange(z_min, z_max, 0.001)
            # calculate P_z = P(z_domain)
            vectorization_integrated_pzfxn = np.vectorize(integrated_pzfxn)
            P_z = vectorization_integrated_pzfxn(z_domain,pzfxn)
        
            uniform_deviate = np.random.uniform(P_z.min(),P_z.max(),ngals)
            z_best = interp1d(P_z,z_domain,kind='linear')(uniform_deviate)           
        else:
            z_best = np.zeros(ngals)+self.config['src_z']
        
        
        if is_zerr: 
            # introduce a redshift error on the source redshifts and generate 
            # the corresponding Gaussian pdf
            self.config['redshift_err'] = sigma_z_ref
            sigma_z = sigma_z_ref*(1.+z_best)
            z_true = z_best + sigma_z*np.random.standard_normal(ngals)
            pdf_grid = []
            zbins_grid = []
            for i,z in enumerate(z_best):
                zmin = z - 0.5
                zmax = z + 0.5
                zbins = np.arange(zmin, zmax, 0.03)
                pdf_grid.append(np.exp(-0.5*((zbins - z)/sigma_z[i])**2)/np.sqrt(2*np.pi*sigma_z[i]**2))
                zbins_grid.append(zbins)
        else:
            # No redshift error
            z_true = z_best

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

        gamt = testProf.deltaSigma(r_mpc)/testProf.Sc(z_true)
        if is_shapenoise:
            self.config['shape_noise'] = shapenoise
            gamt = gamt + shapenoise*np.random.standard_normal(ngals)

        posangle = np.arctan2(y_mpc, x_mpc)
        cos2phi = np.cos(2*posangle)
        sin2phi = np.sin(2*posangle)

        e1 = -gamt*cos2phi
        e2 = -gamt*sin2phi

        if is_zerr:
            self.catalog = Table([seqnr, -x_deg, y_deg, e1, e2, z_best, pdf_grid, zbins_grid], \
                                  names=('id', 'ra','dec','gamma1','gamma2', 'z', 'z_pdf', 'z_bins'))
        else:
            self.catalog = Table([seqnr, -x_deg, y_deg, e1, e2, z_best], \
                                names=('id', 'ra','dec','gamma1','gamma2', 'z'))
