import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u
import pickle
import polaraveraging

cosmo = FlatLambdaCDM(70., Om0 = 0.3) # astropy cosmology setting, will be replaced by ccl

filename = open('test_cl.p', 'rb')
data = pickle.load(filename)
ra_l = data['ra']
dec_l = data['dec']
print ('cluster ra, dec:', ra_l, dec_l)

z = data['z']
print ('cluster redshift:', z)
galcat = data['galcat']
e1 = galcat['e1']
e2 = galcat['e2']
print ('number of source galaxies:', len(e1) )

theta, g_t , g_x = compute_shear(ra_l, dec_l, ra_s, dec_s, e1, e2, sky = "flat")  #calculate distance and tangential shear and cross shear for each source galaxy
#theta, g_t , g_x = compute_shear(ra_l, dec_l, ra_s, dec_s, e1, e2, sky = "curved") #curved sky
rMpc = theta * cosmo.angular_diameter_distance(z).value   # transfer radian to Mpc distance unit

r, gt_proflie, gterr_proflie = make_shear_profile(rMpc, g_t)
r, gx_proflie, gxerr_proflie = make_shear_profile(rMpc, g_x)


plot_profiles(r, gt_proflie, gterr_proflie,gx_proflie,gxerr_proflie, "Mpc")

bins =make_bins(0.1, 3.7,20).   #make new binning range
print (bins)

r, gt_proflie, gterr_proflie = make_shear_profile(rMpc, g_t, bins=bins)
r, gx_proflie, gxerr_proflie = make_shear_profile(rMpc, g_x, bins=bins)

plot_profiles(r, gt_proflie, gterr_proflie,gx_proflie,gxerr_proflie, "Mpc")


plt.errorbar(r,gx_proflie, gxerr_proflie)
plt.title('cross shear test')
plt.ylim(-0.002,0.002)
plt.hlines(0.,np.min(r), np.max(r))
plt.xlabel("r ["+ r_units +"]")
plt.ylabel('$\\gamma$');
plt.show()
