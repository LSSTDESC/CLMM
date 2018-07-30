#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
# model from Dallas group
import colossus.cosmology.cosmology as Cosmology # used for distances
import CLMM_densityModels_beforeConvertFromPerH as clmm
# the code we want to test:
from shear_azimuthal_averager import ShearAzimuthalAverager



def test_shear_azimuthal_averager():
    # compute the profile using our code

    # take an ideal catalog made by Celine
    cl_dict = {'z':0.3, 'ra':0, 'dec': 0, 'm':1e15, 'c':4}

    filename='/global/u1/c/ccombet/shearfile.hdf5'

    atab = Table.read(filename, path='deepCoadd_meas')
    ztab = Table.read(filename, path='zphot_ref')
    t = Table([atab['id'],atab['coord_ra_deg'],atab['coord_dec_deg'],
                atab['ext_shapeHSM_HsmShapeRegauss_e1'],atab['ext_shapeHSM_HsmShapeRegauss_e2'],
                ztab['Z_BEST'],atab['id']*0], names=('id','ra','dec', 'gamma1', 'gamma2', 'z', 'kappa'))

    # # plt.scatter(new_tab['ra'], new_tab['dec'], s=2)
    # # plt.show()
    # saa = ShearAzimuthalAverager(cl_dict=cl_dict,src_table=new_tab)
    # saa.compute_shear()
    # binned_profile = saa.make_shear_profile()


    # # get the galaxies around it
    # ra_min, ra_max = ra_cl-0.3, ra_cl+0.3
    # dec_min, dec_max = dec_cl-0.3, dec_cl+0.3
    # z_min = z_cl + 0.1
    # z_max = 1.5

    # coord_filters = [
    #     'ra >= {}'.format(ra_min),
    #     'ra < {}'.format(ra_max),
    #     'dec >= {}'.format(dec_min),
    #     'dec < {}'.format(dec_max),
    # ]
    # z_filters = ['redshift >= {}'.format(z_min),'redshift < {}'.format(z_max)]

    # gal_cat = extragalactic_cat.get_quantities(['galaxy_id', 'ra', 'dec', 'shear_1', 'shear_2', 'shear_2_phosim', 'shear_2_treecorr',\
    #                                             'redshift', 'convergence'], filters=(coord_filters + z_filters))

    # # store the results into an astropy table
    # t = Table([gal_cat['galaxy_id'],gal_cat['ra'],gal_cat['dec'],gal_cat['shear_1'],\
    #           gal_cat['shear_2'],gal_cat['redshift'],gal_cat['convergence']], \
    #           names=('id','ra','dec', 'gamma1', 'gamma2', 'z', 'kappa'))


    # create an object, given cluster dictionary and galaxy astropy table
    saa = ShearAzimuthalAverager(cl_dict,t)

    # compute tangential and cross shear for each galaxy
    saa.compute_shear()

    # make the binned profile
    binned_profile = saa.make_shear_profile()



    # set up a model
    zL = cl_dict['z'] # NOTE!
    chooseCosmology = 'WMAP7-ML' #Choose cosmology used
    cosmo = Cosmology.setCosmology(chooseCosmology)

    mdef = '200c'
    M = cl_dict['m']*cosmo.h
    c = cl_dict['c']
    r = np.linspace(0.5, 3., 100) #Mpc
    r = r*cosmo.h #Mpc/h


    testProf = clmm.nfwProfile(M = M, c = c, zL = zL, mdef = mdef, \
                            chooseCosmology = chooseCosmology, esp = None)


    z_infty = 300
    zL_arr = np.zeros(len(t['z'])) + zL
    D_LS = (cosmo.comovingDistance(z_min=0.,z_max=t['z']) - cosmo.comovingDistance(z_min=0.,z_max=zL))/(1.+t['z'])
    D_Linfty = (cosmo.comovingDistance(z_min=0., z_max=z_infty) - cosmo.comovingDistance(z_min=0., z_max=zL))/(1.+z_infty)
    beta = D_LS * cosmo.angularDiameterDistance(z_infty)/cosmo.angularDiameterDistance(t['z'])/D_Linfty


    r_phys = binned_profile['radius']
    gt_mod = np.zeros(len(r_phys))

    for ir, r in enumerate(r_phys):
        gamtest_infty= testProf.deltaSigma(np.array([r])*cosmo.h)/testProf.Sc(z_infty)
        kappa_infty =  testProf.Sigma(np.array([r])*cosmo.h)/testProf.Sc(z_infty)
        gt_mod[ir] = np.mean(beta * gamtest_infty/(1-beta*kappa_infty))


    #from scipy.interpolate import interp1d
    #gt_mod_interp = interp1d(r_phys, gt_mod)

    nr = len(binned_profile['g_t']) * 1.
    # equivalent to chi_sqr/nbins
    g_t_residual = sum(abs(binned_profile['g_t'] - gt_mod)/binned_profile['g_t_err'])/nr
    g_x_residual = sum(abs(binned_profile['g_x'])/binned_profile['g_x_err'])/nr

    print(g_t_residual)
    print(g_x_residual)

    tolerance = 0.1 # 
    assert g_t_residual < tolerance
    assert g_x_residual < tolerance

    plotting = True
    if plotting==True:
        plt.plot(r_phys, gt_mod, label='model')
        plt.errorbar(r_phys, binned_profile['g_t'], yerr=binned_profile['g_t_err'], label='data')
        plt.legend()
        plt.show()



test_shear_azimuthal_averager()