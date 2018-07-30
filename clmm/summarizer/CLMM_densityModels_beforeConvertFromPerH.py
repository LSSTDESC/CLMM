# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:05:13 2015

@author: Matthew Fong
"""


import colossus.cosmology.cosmology as Cosmology
from colossus.utils import constants
import colossus.halo.profile_dk14 as profile_dk14
#import HaloDensityProfile
import colossus.halo as Halo
import colossus.halo.concentration as hc
#import HaloConcentration as hc
#import cosmolopy.distance as cd


import numpy as np
from scipy import integrate
from abc import ABCMeta


'''
#More flexible SigmaCrit function
class weakLens(object):
    def __init__(self, zd, cosmoName):
        self.zd = zd
        self.cosmo = Cosmology.setCosmology(cosmoName)
        
        cosmo = {'omega_M_0' : self.cosmo.Om0, \
                 'omega_lambda_0' : self.cosmo.OL0, \
                 'h' : self.cosmo.h}
        self.cosmoDist = cd.set_omega_k_0(cosmo)
    def beta(self, zs):
        zd = self.zd
        Dd = cd.angular_diameter_distance(zd, **self.cosmoDist)
        if isinstance(zs,np.ndarray):
            f = np.piecewise(zs, [zs<=zd, zs>zd], \
                                [lambda zs: 0., \
                                 lambda zs: 1.-Dd*(1.+zd)/cd.comoving_distance(zs, z0=0., **self.cosmoDist)])
        else: 
            if (zs<=self.zd): 
                f = 0.
            else:
                f = 1.-Dd*(1.+zd)/cd.comoving_distance(zs, z0=0., **self.cosmoDist)
        return f
        
    def sigmaC(self, zs):
        zd = self.zd
        Dd = cd.angular_diameter_distance(zd, **self.cosmoDist)
        # some astrophysical parameters
        Mpc2meter=3.08568025*10**22 # Mpc to meters
        Msolar2kg=1.989*10**30 # kg
        clight=2.99792*10**8/Mpc2meter # m/s -> Mpc/s
        G = 6.67428*10**(-11)/(Mpc2meter**3)*Msolar2kg # m3/(kg·s^2) --> Mpc^3/(solar mass * s^2)
        return clight**2/(4*np.pi*G*Dd*self.beta(zs))
'''
class profile(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self, zL, mdef, chooseCosmology):
        self.chooseCosmology = chooseCosmology
        self.zL = zL
        self.mdef = mdef
        self.G, self.v_c, self.H2, self.cosmo = self.calcConstants()
        return
    
    '''
    ############################################################################
                            Constants for Profiles
    ############################################################################
    '''
    

    def calcConstants(self):
        chooseCosmology = self.chooseCosmology
        zL = self.zL
        listCosmologies = ['planck15-only', 'planck15', 'planck13-only', \
    'planck13', 'WMAP9-only', 'WMAP9-ML', 'WMAP9', 'WMAP7-only', 'WMAP7-ML', \
    'WMAP7', 'WMAP5-only', 'WMAP5-ML', 'WMAP5', 'WMAP3-ML', 'WMAP3', \
    'WMAP1-ML', 'WMAP1', 'bolshoi', 'millennium', 'powerlaw']        
        if chooseCosmology is None:
            raise Exception('A name for the cosmology must be set.')
        if chooseCosmology not in listCosmologies:
            msg = 'Cosmology must be one of ' + str(listCosmologies)    
            raise Exception(msg)
        if chooseCosmology in listCosmologies:
            cosmo = Cosmology.setCosmology(chooseCosmology)
        # Gravitational constant [G] = Mpc * m^2 / M_{\odot}/s^2 from kpc * km^2 / M_{\odot} / s^2
        #G = Cosmology.AST_G * 1E3
        G = constants.G * 1E3
        
        # Hubble parameter H(z)^2 [H0] = m/s/Mpc #from km/s/Mpc
        #[H2] = (m/s/Mpc)^2
        H2 = (cosmo.Hz(zL)**2.)*1E6
        # speed of light v_c [v_c] = m/s from [AST_c] = cm/s 
        #v_c = Cosmology.AST_c / 1E2
        v_c = constants.C/1E2
        
        return G, v_c, H2, cosmo
    
    '''
    ############################################################################
                          Critical Surface Mass Density
    ############################################################################
    '''
    
    def Sc(self, zS):
        # [D] = Mpc/h
        Dl = self.cosmo.angularDiameterDistance(self.zL)
        Ds = self.cosmo.angularDiameterDistance(zS) 
        Dls = Ds - (1. + self.zL) * Dl /(1.+zS) #assuming flat cosmology
        ret = (self.v_c)**2. / (4.0 * np.pi * self.G) * Ds / ( Dl * Dls)
        #ret = ret/((1.+self.zL)**2.) # from the use of comoving scale
        #[Sc] = M_dot / Mpc^2 from M{dot} h / Mpc^2
        return ret * self.cosmo.h


    '''
    ############################################################################
                                  rho & Sigma
    ############################################################################
    '''
    
    def rho(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            rho = self.nfwrho(r)
            
        if self.profile == 'dk':
            rho = self.dkrho(r)
            
        if self.profile == 'nfwBMO':
            rho = self.bmorho(r)
            
        return rho
    
    def Sigma(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            Sigma = self.nfwSigma(r)
            
        if self.profile == 'dk':
            Sigma = self.dkSigma(r)
            
        if self.profile == 'nfwBMO':
            Sigma = self.bmoSigma(r)
            
        return Sigma
    
    
    '''
    ############################################################################
                                  SigmaMean
    ############################################################################
    '''
    
    def SigmaMean(self,r):
        #[r] = Mpc/h
        
        if self.profile == 'nfw':
            SigmaMean = self.nfwSigmaMean(r)
        #set first bin of SigmaMean[0] to Sigma[0]
        
        if self.profile == 'dk':
            R = np.asarray(r)
            add = np.arange(0.0001, R[0], 0.001)
            r = []
            r.extend(add)
            r.extend(R)
            r = np.asarray(r)
            
            Sigma = self.Sigma(r)
            SigmaInt = integrate.cumtrapz(Sigma*r, r, initial = 0)
            SigmaMean = 2.*SigmaInt/(r**2.)
            SigmaMean = SigmaMean[len(add):]
            #SigmaMean[0] = Sigma[0]
            
        if self.profile == 'nfwBMO':
            SigmaMean = self.bmoSigmaMean(r)
            
        #SigmaMean[0] = self.Sigma(r[0])
        return SigmaMean
    
    def deltaSigma(self,r):
        #[r] = Mpc/h
        if self.profile == 'nfw':
            dSig = self.nfwDeltaSigma(r)
        if self.profile == 'dk':
            '''
            R = np.asarray(r)
            add = np.arange(0.0001, R[0], 0.001)
            r = []
            r.extend(add)
            r.extend(R)
            r = np.asarray(r)
            
            Sigma = self.Sigma(r)
            SigmaInt = integrate.cumtrapz(Sigma*r, r, initial = 0)
            SigmaMean = 2.*SigmaInt/(r**2.)
            
            Sigma = Sigma[len(add):]
            SigmaMean = SigmaMean[len(add):]
            dSig = SigmaMean - Sigma
            '''
            dSig = self.dkDeltaSigma(r)
        if self.profile == 'nfwBMO':
            dSig = self.bmoSigmaMean(r) - self.bmoSigma(r)
        return dSig
        

'''
############################################################################
                                  NFW
############################################################################
'''
class nfwProfile(profile):
    ##### We're going to swap out ``profile'' with ``Profile1D''
    def __init__(self, M , c , zL, mdef, chooseCosmology, esp = None):
        profile.__init__(self, zL, mdef, chooseCosmology)
        
        #self.parameters = parameters
        self.M_mdef = M #M200 input in M_dot/h
        
        self.c = c
        self.zL = zL
        self.mdef = mdef
        self.chooseCosmology = chooseCosmology
        self.G, self.v_c, self.H2, self.cosmo = self.calcConstants()
        self.r_mdef = Halo.mass_so.M_to_R(self.M_mdef, self.zL, self.mdef)/1E3 #Mpc/h
        self.Delta = int(mdef[:-1])
        #[rho_mdef] = M_dot Mpc^3 from M_{\odot}h^2/kpc^3
        self.rho_mdef = (Halo.mass_so.densityThreshold(self.zL, self.mdef) * 1E9 *(self.cosmo.h)**2.)/self.Delta
        self.rs = self.r_mdef/self.c #Mpc/h
        self.profile = 'nfw'
        if esp == None:
            self.esp = 1E-5
        else:
            self.esp = esp
        return
        
    '''
    ############################################################################
                               Analytic NFW profile
    ############################################################################
    '''
    def charOverdensity(self):
        Delta = int(self.mdef[:-1])
        sigma_c = (Delta/3.)*(self.c**3.)/(np.log(1. + self.c) - self.c/(1. + self.c))
        return sigma_c #unitless
    
    def nfwrho(self, R):
        #R in Mpc/h
        #[sigma_c] = unitless
        #[rho_mdef] = M_dot / Mpc^3
        
        const =  self.rho_mdef * self.charOverdensity() 
        rhoForm = 1./( (R/self.rs) * (1. + R/self.rs)**2.)
        return (const * rhoForm)
        
    def nfwSigma(self, r):
        #[r] = Mpc/h
        
        #[rs] = Mpc/h        
        rs = self.rs
        expSig = np.empty(len(r))
        for i in range(len(r)):
            if r[i]/self.rs < 1.0 - self.esp:
                expSig[i] = (1./((r[i]/rs)**2. - 1.))*(1. - 2.*np.arctanh(np.sqrt(\
                (1. - (r[i]/rs))/(1. + (r[i]/rs))))/np.sqrt(1. - (r[i]/rs)**2.))
            #if r[i]/rs == 1.0:
            #    expSig[i] = 1./3.
            if r[i]/rs >= 1.0 - self.esp and r[i]/rs <= 1.0 + self.esp:
                expSig[i] = 1./3.
            if r[i]/self.rs > 1.0 + self.esp:
                expSig[i] = (1./((r[i]/rs)**2. - 1.))*(1. - 2.*np.arctan(np.sqrt(\
                ((r[i]/rs) - 1.)/(1. + (r[i]/rs))))/np.sqrt((r[i]/rs)**2. - 1.))
        const = 2.*(self.rs)*self.charOverdensity()*(self.rho_mdef)
        #[Sigma] = M_dot / Mpc^2 from M{dot} / h / Mpc^2
        return (expSig * const)/self.cosmo.h 
    
    def nfwSigmaMean(self, r):
        #[r] = Mpc/h
        x = r/self.rs
        const = 4.*self.rs*self.charOverdensity()*self.rho_mdef
        if type(x) is np.ndarray:
            #print x
            fnfw = np.piecewise(x,[x<1., x==1., x>1.], \
                        [lambda x:(2./np.sqrt(1.-x*x)*np.arctanh(np.sqrt((1.-x)/(1.+x)))+np.log(x/2.))/(x*x), \
                         lambda x:(1.+np.log(0.5)), \
                         lambda x:(2./np.sqrt(x*x-1.)*np.arctan2(np.sqrt(x-1.),np.sqrt(1.+x))+np.log(x/2.))/(x*x)])
            return const*fnfw
    
        else:
            if x<1:
                return const*(2./np.sqrt(1.-x*x)*np.arctanh(np.sqrt((1.-x)/(1.+x)))+np.log(x/2.))/(x*x)
            elif x==1:
                return const*(1.+np.log(0.5))
            else:
                return const*(2./np.sqrt(x*x-1.)*np.arctan2(np.sqrt(x-1.),np.sqrt(1.+x))+np.log(x/2.))/(x*x)
        
        
    
    def nfwDeltaSigma(self, r):
        #[r] = Mpc/h
        rs = self.rs
        
        x = r/rs
        expG = np.empty(len(x))
        for i in range(len(x)):
            if x[i] < 1.0 - self.esp:
                expG[i] = 8.0*np.arctanh(np.sqrt((1.0-(x[i]))/(1.0+(x[i]))))/(((x[i])**2.0)*np.sqrt(1.0-(x[i])**2.0))\
                + (4.0/((x[i])**2.0))*np.log((x[i])/2.0) \
                - 2.0/((x[i])**2.0 - 1.0)\
                + 4.0*np.arctanh(np.sqrt((1.0-(x[i]))/(1.0+(x[i]))))/(((x[i])**2.0 - 1.0)*np.sqrt(1.0-(x[i])**2.0))
            if x[i] > 1.0 - self.esp and x[i] < 1.0 + self.esp:
                expG[i] = (10.0 / 3.0) + (4.0 * np.log(1.0/2.0))
            if x[i] > 1.0 + self.esp:
                expG[i] = 8.0*np.arctan(np.sqrt(((x[i])-1.0)/(1.0+(x[i]))))/(((x[i])**2.0)*np.sqrt((x[i])**2.0 - 1.0))\
                + (4.0/((x[i])**2.0))*np.log((x[i])/2.0) \
                - 2.0/((x[i])**2.0 - 1.0)\
                + 4.0*np.arctan(np.sqrt(((x[i])-1.0)/(1.0+(x[i]))))/(((x[i])**2.0 - 1.0)**(3.0/2.0))
        
        
        #[rho_mdef] = M_dot/Mpc^3 
        #[charOverdensity] = unitless
        charOverdensity = self.charOverdensity()
        Const = (rs) * (self.rho_mdef) * (charOverdensity)
        # [Const] = 1 from h^-1
        Const = Const/(self.cosmo.h)
        return Const*expG
    
        
        
    
'''
############################################################################
                               Truncated NFW
############################################################################
'''
class nfwBMOProfile(profile):
    
    def __init__(self, parameters, zL, n, mdef, chooseCosmology, Tau = None, cM_relation = None, esp = None):
        profile.__init__(self, zL, mdef, chooseCosmology)
        
        cosmo = Cosmology.setCosmology(chooseCosmology)
        self.parameters = parameters
        self.M_mdef = parameters['M'].value #M200 input in M_dot/h
        if cM_relation == True:
            self.c = hc.concentration(self.M_mdef*cosmo.h, self.mdef, self.zL)
        else:
            self.c = parameters['c'].value
        self.zL = zL
        self.n = n #sharpness of truncation (n = 1 or 2)
        if Tau == None: #dimensionless truncation radius (T = rt/rvir => fit =2.6)
            self.T = 2.6
        else:
            self.T = Tau
        self.r_mdef = Halo.mass_so.M_to_R(self.M, self.zL, self.mdef)/1E3 #Mpc/h
        self.rs = self.r_mdef/self.c
        self.rt = self.T*self.rs
        self.mdef = mdef
        self.chooseCosmology = chooseCosmology
        self.G, self.v_c, self.H2, self.cosmo = self.calcConstants()
        self.Delta = int(mdef[:-1])
        #[rho_mdef] = M_dot Mpc^3 from M_{\odot}h^2/kpc^3
        self.rho_mdef = (Halo.mass_so.densityThreshold(self.zL, self.mdef) * 1E9 *(self.cosmo.h)**2.)/self.Delta
        
        self.profile = 'nfwBMO'
        if esp == None:
            self.esp = 1E-5
        else:
            self.esp = esp
        
        return
        #super(nfwProfile, self).__init__()

        
    '''
    ############################################################################
                               Analytic BMO profile
    ############################################################################
    '''
    def mnfw(self):
        m = (np.log(1. + self.c) - self.c/(1. + self.c))
        return m

    def densityParameter(self):
        rho_s = self.M/(4.*np.pi*(self.rs**3.)*self.mnfw())
        #[rho_s] = M_dot / Mpc^3 from M_dot h^2 / Mpc^3
        rho_s = rho_s * (self.cosmo.h**2.)
        #[rho_s] = M_dot / Mpc^3
        return rho_s
        
    def bmorho(self, R):
        #R in Mpc/h
        #[rho_s] = M_dot / Mpc^3
        rho_s = self.densityParameter()
        rho_nfw = (rho_s/((R/self.rs)*(1.+R/self.rs)**2.))*((self.rt**2.)/(R**2. + self.rt**2.))**self.n
        #[rho_nfw] = M_dot / Mpc^3
        return rho_nfw
    
    def F(self, r):
        #[r] = Mpc/h
        func = np.empty(len(r))
        x = r/self.rs
        for i in range(len(x)):
            if x[i] < 1.0:
                func[i] = (1./(np.sqrt(1. - x[i]**2.)))*np.arctanh(np.sqrt(1.-x[i]**2.))
            if x[i] > 1.0:
                func[i] = (1./(np.sqrt(x[i]**2. - 1.)))*np.arctan(np.sqrt(x[i]**2.-1.))
        return func
    
    def L(self, r):
        #[r] = Mpc/h
        x = r/self.rs
        T = self.T
        func = np.log(x/(np.sqrt(T**2. + x**2.) + T))
        return func
        
    def bmoSigma(self, r):
        #[r] = Mpc/h
        x = r/self.rs
        T = self.T
        #[densityParameter] = M_dot / Mpc^3
        #[Const] = M_dot/Mpc^2 from M_dot / h / Mpc^2
        Const = (4.*self.densityParameter()*self.rs)*((T**2.)/(2.*(T**2. + 1.)**2.))
        Const = Const/self.cosmo.h
        if self.n == 1:
            func = ((T**2. + 1.)/(x**2. - 1.))*(1.-self.F(r)) + 2.*self.F(r)\
                    - np.pi/(np.sqrt(T**2. + x**2.))\
                    +((T**2. - 1.)/(T*(np.sqrt(T**2. + x**2.))))*self.L(r)
        if self.n == 2:
            Const = Const*(T**2./(2.*(T**2.+1.)))
            func = ((2.*(T**2. + 1.))/(x**2. - 1.))*(1. - self.F(r))\
                    + 8.*self.F(r) + (T**4. - 1.)/((T**2.)*(T**2. + x**2.))\
                    - np.pi*((4.*(T**2. + x**2.)\
                    + T**2. + 1.)/((T**2. + x**2.)**(3./2.)))\
                    + (((T**2.)*(T**4. - 1.) + (T**2. + x**2.)*(3.*T**4. - 6.*T**2. - 1.))\
                    / ((T**3.)*(T**2. + x**2.)**(3./2.)))*self.L(r)
        #[Sigma] = M_dot/Mpc^2
        return Const*func
        
        
    def bmoSigmaMean(self, r):
        x = r/self.rs
        T = self.T
        Const = (4.*self.densityParameter()*self.rs)
        #[Const] = M_dot / Mpc^2 from M_dot / h / Mpc^2
        Const = Const / self.cosmo.h
        if self.n == 1:
            func = ((T**2.)/((x**2.)*(T**2. + 1.)**2.))\
                    *(\
                    (T**2. + 2.*x**2. + 1.)*self.F(r)\
                    + T*np.pi + (T**2. - 1.)*np.log(T)\
                    + np.sqrt(T**2. + x**2.)*(-np.pi + ((T**2. - 1.)/T)*self.L(r))\
                    )
        if self.n == 2:
            func = ((T**4.)/(2.*(x**2.)*(T**2. + 1.)**3.))\
                    *(2.*(T**2. + 4.*x**2. - 3.)*self.F(r)\
                    +(1./T)*(\
                    np.pi*(3.*T**2. - 1.)\
                    + 2.*T*(T**2. - 3.)*np.log(T)\
                    )\
                    + (1./((T**3.)*(np.sqrt(T**2. + x**2.))))\
                    *(\
                    (-(T**3.))*np.pi*(4.*x**2. + 3.*T**2. - 1.)\
                    + ((2.*T**4.)*(T**2. - 3.)\
                    + (x**2.)*(3.*T**4. - 6.*T**2. - 1.))*self.L(r)\
                    ))
        #[SigmaMean] = M_dot / Mpc^2
        return Const*func   



'''
############################################################################
                             Numerical dk14
############################################################################
'''

class dkProfile(profile):
    
    def __init__(self, parameters, zL, mdef, chooseCosmology, part = None, \
                 se = None, be = None, cM_relation = None):
        profile.__init__(self, zL, mdef, chooseCosmology)
        
        
        self.M_mdef = parameters['M'].value #M200 in M_dot/h
        
        if cM_relation == True:
            self.c = hc.concentration(self.M, self.mdef, self.zL)
            #self.c = 3.614*((1+self.zL)**(-0.424))*(self.M/self.cosmo.h/1E14)**(-0.105)
        else:
            self.c = parameters['c'].value
        
        if se is None:
            self.se = 1.5
        else:
            self.se = se
        if be is None:
            self.be = 1.0
        else:
            self.be = be
        
        
        self.zL = zL
        self.mdef = mdef
        if part is not None:
            self.part = part
        else:
            self.part = 'both'
        #[rs] = Mpc/h
        self.r_mdef = Halo.mass_so.M_to_R(self.M_mdef, self.zL, self.mdef)/1E3 #Mpc/h
        self.rs = self.r_mdef/self.c #Mpc/h
        self.Delta = int(mdef[:-1])
        #[rho_mdef] = M_dot Mpc^3 from M_{\odot}h^2/kpc^3
        self.rho_mdef = (Halo.mass_so.densityThreshold(self.zL, self.mdef) * 1E9 *(self.cosmo.h)**2.)/self.Delta
        
        '''
        self.dk14Prof = HaloDensityProfile.DK14Profile(M = self.M_mdef, c = self.c, z = \
                                                       self.zL, mdef = self.mdef, \
                                                       be = self.be, se = self.se, \
                                                       part = self.part)
        '''
        self.dk14Prof = profile_dk14.getDK14ProfileWithOuterTerms(M = self.M_mdef, c = self.c, z = self.zL, 
                                     mdef = self.mdef, 
                                     outer_term_names = ['pl'])
        #self.dk14Prof.par.se = self.se
        #self.dk14Prof.par.be = self.be
        self.rmaxMult = 2.
        
        #self.dk14Prof.par.rs = self.rs*1E3 #[rs] = kpc/h from Mpc/h
        #self.dk14Prof.selected = 'by_accretion_rate' #beta = 6 gamma = 4; more accurate results
        self.dk14Prof.selected = 'by_mass' 
        self.profile = 'dk'
        #super(dkProfile, self).__init__()
        
        return
    
    def dkrho(self,R):
        #input [R] = Mpc/h
        R = R*1E3 #[R] = kpc/h from Mpc/h for Diemer input
        rho = self.dk14Prof.density(R) *1E9 #[rho] = M_dot h^2 / Mpc^3 from M_{\odot} h^2/ kpc^3
        #[rho] = M_dot / Mpc^3 from M_dot h^2 / Mpc^3
        rho = rho * (self.cosmo.h**2.)
        return rho 
    
    def dkSigma(self,R):
        #input [R] = Mpc/h
        #[R] = kpc/h from Mpc/h for Diemer input
        r = R*1E3
        self.dk14Prof.rmax = r[-1]*self.rmaxMult
        #[surfaceDensity] = M_dot h/Mpc^2 from M_{\odot} h/kpc^2
        SigmaDiemer = self.dk14Prof.surfaceDensity(r) * 1E6
        #[Sigma] = M_dot / Mpc^2 from `M_{\odot} h/Mpc^2`
        SigmaDiemer = SigmaDiemer * self.cosmo.h
        return SigmaDiemer #[Sigma] = M_dot/Mpc^2
    
    def dkDeltaSigma(self,r):
        #input [r] = Mpc/h
        '''
        #[R] = kpc/h from Mpc/h for Diemer input
        r = R*1E3
        self.dk14Prof.rmax = r[-1]*self.rmaxMult
        #[surfaceDensity] = M_dot h/Mpc^2 from M_{\odot} h/kpc^2
        SigmaDiemer = self.dk14Prof.surfaceDensity(r) * 1E6
        #[Sigma] = M_dot / Mpc^2 from `M_{\odot} h/Mpc^2`
        SigmaDiemer = SigmaDiemer * self.cosmo.h
        '''
        #[r] = kpc/h from Mpc (Diemer input)
        dSig = self.dk14Prof.deltaSigma(r*1E3*self.cosmo.h)
        #output :math:`M_{\odot} h/{\\rm kpc}^2`
        
        return dSig * self.cosmo.h #[Sigma] = M_dot/Mpc^2 from M_dot h^2 / Mpc^2
    
    


