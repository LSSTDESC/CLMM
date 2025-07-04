import Magneticum_maps_KDTree_all
import sys
import os
import yaml

# Read YAML config file path from command line
config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Load inputs from YAML
snap_num   = int(config['snapshot'])       # snapshot number (redshift)
cosmo_name = config['cosmo_name']          # cosmology name (C1–C14)
sim_type   = config['sim_type']            # 'hydro' or 'dm'
l_proj     = config['l_proj']              # half-length of projection cylinder
r_max      = config['r_max']               # radius of projection cylinder
res        = config['resolution']          # map resolution


# define cosmologies
cosmologies = {}
cosmologies['C1']  = '0.153_0.0408_0.614_0.666'
cosmologies['C2']  = '0.189_0.0455_0.697_0.703'
cosmologies['C3']  = '0.200_0.0415_0.850_0.730'
cosmologies['C4']  = '0.204_0.0437_0.739_0.689'
cosmologies['C5']  = '0.222_0.0421_0.793_0.676'
cosmologies['C6']  = '0.232_0.0413_0.687_0.670'
cosmologies['C7']  = '0.268_0.0449_0.721_0.699'
cosmologies['C8']  = '0.272_0.0456_0.809_0.704'
cosmologies['C9']  = '0.301_0.0460_0.824_0.707'
cosmologies['C10'] = '0.304_0.0504_0.886_0.740'
cosmologies['C11'] = '0.342_0.0462_0.834_0.708'
cosmologies['C12'] = '0.363_0.0490_0.884_0.729'
cosmologies['C13'] = '0.400_0.0485_0.650_0.675'
cosmologies['C14'] = '0.406_0.0466_0.867_0.712'
cosmologies['C15'] = '0.428_0.0492_0.830_0.732'

# select cosmology
cosmo = cosmologies[cosmo_name]


# input dir for DM and hydro catalogs
sim_base_hydro = "/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box1a/mr_%s/" %cosmo
sim_base_dm    = "/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/Box1a/mr_dm_%s/" %cosmo

# output file
cat_file     = "/gpfs/scratch/uh101/ra37duv2/massmaps/halo_catalog_%s_snap%02d.h5py" %(cosmo,snap_num)


# if output catalog already existing, no need for creating a new one
if os.path.exists(cat_file):
    print("Catalog already exists")
    do_catalog=False
else:
    print("Catalog to be created")
    do_catalog=True


# run the code to create mass maps from the selected simulation    
Magneticum_maps_KDTree_all.main(cat_file, sim_base_dm, sim_base_hydro, snap_num, sim_type, do_catalog=do_catalog, l_proj=l_proj, r_max=r_max, res=res)
