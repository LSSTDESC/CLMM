import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u
import pickle
import polaraveraging
import clmm

cosmo_object_type = "astropy"
try:
    import pyccl
except:
    cosmo_object_type = "astropy"
if cosmo_object_type == "astropy":
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(70., Om0 = 0.3) # astropy cosmology setting, will be replaced by ccl
elif cosmo_object_type == "ccl":
    import pyccl as ccl
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b = 0.05,h = 0.7,n_s = 0.97, sigma8 = 0.8, Omega_k = 0.)

cl = clmm.load_cluster('9687686568.p')

ra_l = cl.ra
dec_l = cl.dec
z = cl.z

e1 = cl.galcat['e1']
e2 = cl.galcat['e2']

ra_s = cl.galcat['ra']
dec_s = cl.galcat['dec'] 

theta, g_t , g_x = polaraveraging._compute_shear(ra_l, dec_l, ra_s, dec_s, e1, e2, sky = "flat")  #calculate distance and tangential shear and cross shear for each source galaxy
#theta, g_t , g_x = compute_shear(ra_l, dec_l, ra_s, dec_s, e1, e2, sky = "curved") #curved sky
rMpc = polaraveraging._theta_units_conversion(theta,"Mpc",z,cosmo,cosmo_object_type=cosmo_object_type)
r, gt_proflie, gterr_proflie = polaraveraging._compute_radial_averages(rMpc, g_t)
r, gx_proflie, gxerr_proflie = polaraveraging._compute_radial_averages(rMpc, g_x)


polaraveraging._plot_profiles(r, gt_proflie, gterr_proflie,gx_proflie,gxerr_proflie, "Mpc")
plt.show()
bins =polaraveraging.make_bins(0.1, 3.7,20)   #make new binning range
#print (bins)

r, gt_proflie, gterr_proflie = polaraveraging._compute_radial_averages(rMpc, g_t, bins=bins)
r, gx_proflie, gxerr_proflie = polaraveraging._compute_radial_averages(rMpc, g_x, bins=bins)

polaraveraging._plot_profiles(r, gt_proflie, gterr_proflie,gx_proflie,gxerr_proflie, "Mpc")
plt.show()

plt.errorbar(r,gx_proflie, gxerr_proflie)
plt.title('cross shear test')
plt.ylim(-0.002,0.002)
plt.hlines(0.,np.min(r), np.max(r))
plt.xlabel("r")
plt.ylabel('$\\gamma$')
plt.show()
