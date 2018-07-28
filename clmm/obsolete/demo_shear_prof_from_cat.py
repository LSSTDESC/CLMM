import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import FoFCatalogMatching
import GCRCatalogs

extragalactic_cat = GCRCatalogs.load_catalog('proto-dc2_v2.1.2_test')
massive_halos = extragalactic_cat.get_quantities(['halo_mass', 'redshift','ra', 'dec'], filters=['halo_mass > 1e14','is_central==True'])

m = massive_halos['halo_mass']
select = (m == np.max(m))
ra_cl = massive_halos['ra'][select][0]
dec_cl = massive_halos['dec'][select][0]
z_cl = massive_halos['redshift'][select][0]
print(m[select], ra_cl, dec_cl, z_cl)


ra_min, ra_max = ra_cl-0.1, ra_cl+0.1
dec_min, dec_max = dec_cl-0.1, dec_cl+0.1
z_min = z_cl + 0.1
z_max = 1.5

coord_filters = [
    'ra >= {}'.format(ra_min),
    'ra < {}'.format(ra_max),
    'dec >= {}'.format(dec_min),
    'dec < {}'.format(dec_max),
]
z_filters = ['redshift >= {}'.format(z_min),'redshift < {}'.format(z_max)]
gal_cat = extragalactic_cat.get_quantities(['ra', 'dec', 'shear_1', 'shear_2', 'shear_2_phosim', 'shear_2_treecorr','redshift'], filters=(coord_filters + z_filters))


ra = gal_cat['ra']
dec = gal_cat['dec']
shear1 = gal_cat['shear_1']
shear2 = gal_cat['shear_2']
z = gal_cat['redshift']
tmp = plt.hist(shear1, bins=50)




#plt.scatter(shear1, shear2, marker='+')



phi = np.arctan2(dec-dec_cl, ra-ra_cl)
print(len(phi), len(shear1))
gamt = - (shear1 * np.cos(2.0 * phi) - shear2 * np.sin(2.0 * phi))
gamc = - shear1 * np.sin(2.0 * phi) + shear2 * np.cos(2.0 * phi)
print(phi*180./np.pi)



theta = np.sqrt((ra-ra_cl)**2+(dec-dec_cl)**2)*np.pi/180.
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
phys_dist = theta * cosmo.angular_diameter_distance(z).value



#plt.hist(phys_dist, bins=50)



# plt.hist(z)

#lnr_arr = np.linspace(np.log(0.5), np.log(3), 10)
r_arr = np.linspace(0.75,3,10)#np.exp(lnr_arr)
nr = len(r_arr)-1
rmean_arr = np.zeros(nr)
gamt_arr = np.zeros(nr)
gamt_err_arr = np.zeros(nr)
gamc_arr = np.zeros(nr)
for ir in range(nr):
    r_min = r_arr[ir]
    r_max = r_arr[ir+1]
    select = (phys_dist >= r_min) & (phys_dist < r_max)
    rmean_arr[ir] = np.mean(phys_dist[select])
    print(np.mean(gamt[select]), len(gamt[select]))
    gamt_arr[ir] = np.mean(gamt[select])
    gamt_err_arr[ir] = np.std(gamt[select])
    gamc_arr[ir] = np.mean(gamc[select])



#plt.errorbar(rmean_arr, gamt_arr, yerr=gamt_err_arr)
#plt.errorbar(rmean_arr, gamc_arr, yerr=gamt_err_arr)
#plt.xscale('log')
#plt.yscale('log')


