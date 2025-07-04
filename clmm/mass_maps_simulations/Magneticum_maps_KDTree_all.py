import  os
import numpy as np
import g3read
import h5py
import sys
from mpi4py import MPI
import time
from scipy.spatial import KDTree


tags = {'READY': 0,
        'START': 1,
        'DONE': 2,
        'EXIT': 3}

def main(catalog_file, sim_base_dm, sim_base_hydro, snap_num, sim_type, do_catalog=True, l_proj=20e3, r_max=5e3, res=10):
    r"""Make Sigma maps for all halos in the catalog with has_profile=1.

    Parameters
    ----------    
    catalog_file   = .h5py file for saving DM and matched HYDRO halos
    sim_base_dm    = DM original catalog (Magneticum)
    sim_base_hydro = HYDRO original catalog (Magneticum)
    snap_num       = snapshot number 
    sim_type       = simulation type ('dm' or 'hydro') for creating maps
    do_catalog     = if True, creates the catalog_file from original snapshots 
    l_proj         = length of the projection cylinder [kpc/h]
    r_max          = radius of the projection cylinder [kpc/h]
    res            = maps resolution
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    
    # Create the catalog if it doesn't already exist    
    if do_catalog==True:
        if rank==0:
            create_catalog(catalog_file, sim_base_dm, sim_base_hydro, snap_num)
        comm.Barrier() # Ensure all processes wait for rank 0 to finish
        

    # Set up different read parameters depending on whether we are processing DM or hydro simulations
    if sim_type=='dm':
        blocks = ['POS ',]  # Only particle positions are needed for DM-only case
        ptypes = [1, 2]     # Particle types corresponding to dark matter (1, sometimes 2)
        POS_label = 'GPOS'
        sim_base  = sim_base_dm
        leafsize  = 30
    elif sim_type=='hydro':
        blocks = ['POS ', 'MASS', 'BHMA']  # Need mass and BH info in addition to positions
        ptypes = [0, 1, 4, 5]              # Gas, DM, stars, black holes
        POS_label = 'hydro_GPOS'
        sim_base  = sim_base_hydro
        leafsize  = 20    # Lower leafsize for KDTree due to potentially denser particle distribution
    else:
        print("Unkonwn sim_type %s"%sim_type)
        return

    # Snapshot properties, halo names and positions
    if rank==0:
        header = g3read.GadgetFile('%s/snapdir_%s/snap_%s.0'%(sim_base,str(snap_num).zfill(3),str(snap_num).zfill(3))).header
        masses = header.mass
        box_size  = header.BoxSize
        scale_fac = header.time
        print('snapshot %s redshift %.3f %s, %d files'%(snap_num, 1/scale_fac-1, sim_type, header.num_files))
        with h5py.File(catalog_file, 'r') as cat_hdf5:
            # All halo names
            all_names   = list(cat_hdf5['halos'].keys())
            has_profile = np.array([cat_hdf5['halos'][name]['has_profile'][()] for name in all_names])
            good        = (has_profile==1).nonzero()[0]
            names_prof  = [all_names[g] for g in good]
            halo_pos    = [cat_hdf5['halos'][name][POS_label][:] for name in names_prof]
    else:
        box_size, scale_fac, names_prof, halo_pos, masses = 5 * [None]

    box_size   = comm.bcast(box_size, root=0)
    scale_fac  = comm.bcast(scale_fac, root=0)
    names_prof = comm.bcast(names_prof, root=0)
    halo_pos   = comm.bcast(halo_pos, root=0)
    masses     = comm.bcast(masses, root=0)

    # Initialize master
    if rank==0:
        with h5py.File(catalog_file, 'a') as cat_hdf5:
            
            # Store metadata for the projection (depth of projection, resolution of map)
            for name, var in zip(['l_proj', 'resolution [kpc/h]'], [l_proj, res]):
                if name in cat_hdf5.keys():
                    cat_hdf5[name][()] = var
                else: 
                    _ = cat_hdf5.create_dataset(name, data=var)
                    
            # Create empty maps for halos with has_profile==1
            for name in cat_hdf5['halos'].keys():
                if '%s_Sigma'%sim_type in cat_hdf5['halos'][name].keys():
                    del cat_hdf5['halos'][name]['%s_Sigma'%sim_type]
                if cat_hdf5['halos'][name]['has_profile'][()]:
                    N = 2*int(r_max/res)+1
                    _ = cat_hdf5['halos'][name].create_dataset('%s_Sigma'%sim_type, data=np.zeros((3, N-1, N-1)))
                    # This is a 3-view projection (XY, YZ, ZX), centered on each halo
                    # spanning ±r_max in each direction, pixel size ~res
            print('Empty maps written', flush=True)

            file_idx = 0
            workers  = np.arange(1, size)
            while len(workers)>0:
                for worker_id in workers:
                    if comm.Iprobe(source=worker_id, tag=MPI.ANY_TAG):
                        data = comm.recv(source=worker_id, tag=MPI.ANY_TAG, status=status)
                        tag  = status.Get_tag()
                        if tag==tags['READY']:
                            if file_idx<header.num_files:
                                comm.send(file_idx, dest=worker_id, tag=tags['START'])
                                file_idx+= 1
                            else:
                                comm.send(None, dest=worker_id, tag=tags['EXIT'])
                                workers = workers[workers!=worker_id]
                        elif tag==tags['DONE']:
                            cat_hdf5['halos'][data[0]]['%s_Sigma'%sim_type][...]+= data[1]
                time.sleep(1.)
        print('Done')

    else:
        while True:
            comm.send(None, dest=0, tag=tags['READY'])
            file_idx = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag==tags['START']:
                # Read snapshot file
                t = [time.time(),]

                # Read particle data from this snapshot chunk
                file_name = '%s/snapdir_%s/snap_%s.%d'%(sim_base,str(snap_num).zfill(3),str(snap_num).zfill(3), file_idx)
                particles = g3read.read_new(file_name, blocks=blocks, ptypes=ptypes)

                # Construct a KDTree for efficient spatial lookup
                t.append(time.time())
                particle_tree = KDTree(particles['POS ']*scale_fac, boxsize=scale_fac*box_size,leafsize=leafsize)
                t.append(time.time())
                
                # Assign masses to particles
                if sim_type=='dm':
                    particles['MASS'] = masses[particles['PTYPE']]
                else:
                    # Use BH subgrid mass instead of particle mass, if available
                    bh_idx = (particles['PTYPE']==5).nonzero()[0]
                    if len(particles['BHMA'])==len(particles['MASS']):
                        particles['MASS'][bh_idx] = particles['BHMA'][bh_idx]
                    else:
                        particles['MASS'][bh_idx] = particles['BHMA']
                        
                # Build maps for each halo
                N = 0
                for n, name in enumerate(names_prof):
                    this_map = my_utils.particles_to_grid_tree(particle_tree,
                                                               scale_fac*particles['POS '],
                                                               particles['MASS'],
                                                               scale_fac*halo_pos[n],
                                                               r_max,
                                                               l_proj,
                                                               scale_fac*box_size,
                                                               res)
                    if this_map is not None:
                        comm.send([name, this_map], dest=0, tag=tags['DONE'])
                        N+= 1
                t.append(time.time())        
                print('rank', rank, 'file_idx', file_idx, '%d halos done, %d had particles'%(len(names_prof), N), np.diff(t), flush=True)
            elif tag==tags['EXIT']:
                print('rank', rank, 'exiting', flush=True)
                break


def particles_to_grid_tree(tree, pos, mass, center, r_max, l_proj, box_size, res):
    r"""Create mass map by projecting particles around halos into a cylinder of radius r_max and depth l_proj

    Parameters
    ----------    
    tree   = particle tree
    pos    = particle positions
    mass   = particle mass
    center = halo center
    r_max  = radius of the projection cylinder [kpc/h]
    l_proj = length of the projection cylinder [kpc/h]
    res    = map resolution
    """

    # Set up empty map
    N = 2*int(r_max/res)+1
    bins = np.linspace(-(N-1)/2*res, (N-1)/2*res, N)
    maps = np.zeros((3, N-1, N-1))
    
    # Particles within sphere
    r_ball = np.sqrt(l_proj**2 + 2*r_max**2)
    ball_idx = tree.query_ball_point(center, r_ball)
    if len(ball_idx)==0:
        return None
    pos  = pos[ball_idx]
    mass = mass[ball_idx]
    
    # Particles within box
    dist = np.abs(center-pos)
    idx = dist>box_size/2
    dist[idx] = box_size-dist[idx]
    if l_proj>r_max:
        r_cutout = l_proj
    else:
        r_cutout = r_max
        
    cutout_idx = ((dist[:,0]<r_cutout) & (dist[:,1]<r_cutout) & (dist[:,2]<r_cutout)).nonzero()[0]
    if len(cutout_idx)==0:
        return None    
    pos  = pos[cutout_idx]
    mass = mass[cutout_idx]
    
    # 3 cartesian projections
    for proj_id in range(3):
        proj_pl = [0, 1, 2]
        proj_pl.remove(proj_id)
        
        # Center particles on halo
        dist = pos[:,proj_pl]-center[proj_pl]
        idx  = dist>box_size/2
        dist[idx] = dist[idx]-box_size
        idx  = dist<-box_size/2
        dist[idx] = dist[idx]+box_size
        
        # Create map
        maps[proj_id,:,:] = np.histogram2d(dist[:,0], dist[:,1], bins=bins, weights=mass)[0]
        
    if np.all(maps==0.):
        return None
    return maps


# Extracts physical and structural halo properties from raw simulation outputs 
# and saves them in HDF5 for easier downstream access
def dump_catalog(sim_base, snap, output_file):
    r"""Dump catalogs from original snapshots to .h5py files

    Parameters
    ----------    
    sim_base    =  original catalog (Magneticum)
    snap        = snapshot number 
    output_file = .h5py file for saving halos properties
    """

    properties = ['GPOS', 'MCRI', 'M200', 'M500', 'M5CC', 'MVIR', 'RVIR']
    my_prop_name = ['GPOS', 'M200c', 'M200m', 'M500m', 'M500c', 'Mvir', 'Rvir']
    header = 'GPOS_x[kpc*a/h] GPOS_y[kpc*a/h] GPOS_z[kpc*a/h] RCRI [kpc*a/h] MCRI[1e10 Msun/h]'
    Mmin = 1.56e4


    halo_id = 0
    values_subfind_file = {}
    test_file = os.path.join(sim_base, 'groups_%s'%str(snap).zfill(3), 'sub_%s.0'%str(snap).zfill(3))
    header = g3read.GadgetFile(test_file, is_snap=False).header
    with h5py.File(output_file, 'w') as f:
        _ = f.create_dataset('data_path', data=sim_base)
        _ = f.create_dataset('BoxSize', data=header.BoxSize)
        _ = f.create_dataset('redshift', data=header.redshift)
        _ = f.create_dataset('scale_fac', data=header.time)
        g = f.create_group('halos')
        for ifile in range(header.num_files):
            # Read data
            filename = os.path.join(sim_base, 'groups_%s'%str(snap).zfill(3), 'sub_%s.%d'%(str(snap).zfill(3), ifile))
            print (filename)
            s = g3read.GadgetFile(filename, is_snap=False)

            nclusters_in_file = s.header.npart[0]
            # Copy all properties (ordered by prop not halo ID)
            for prop in properties:
                values_subfind_file[prop] = s.read_new(prop, 0)
            # Halos above mass cut
            idx_good = (values_subfind_file['MCRI']>=Mmin).nonzero()[0]
            # For each halo, copy into hdf5
            for icluster_file in idx_good:
                halo_name = halo_id+icluster_file
                gg = g.create_group(str(halo_name))
                for prop,my_prop in zip(properties, my_prop_name):
                    _ = gg.create_dataset(my_prop, data=values_subfind_file[prop][icluster_file])
            halo_id+= nclusters_in_file

    print("Read %d files" %ifile)


# Matches halos between DM-only and HYDRO simulations 
def match_catalogs(cat_dm, cat_hydro):
    r"""Match DM and HYDRO halos

    Parameters
    ----------    
    cat_dm    =  DM catalog (.h5py file)
    cat_hydro =  HYDRO catalog (.h5py file)

    """  
    
    with h5py.File(cat_dm, 'a') as f_dm:
        box_size = f_dm['BoxSize'][()]
        with h5py.File(cat_hydro, 'r') as f_hydro:
            all_pos_dm  = np.array([f_dm['halos'][name]['GPOS'] for name in f_dm['halos'].keys()])
            all_pos_hydro = np.array([f_hydro['halos'][name]['GPOS'] for name in f_hydro['halos'].keys()])
            names_hydro   = [name for name in f_hydro['halos'].keys()]
            # Distances between all halo pairs
            diff_pos = np.abs(all_pos_dm[:,None] - all_pos_hydro[None,:])
            idx      = diff_pos>box_size/2
            diff_pos[idx] = box_size-diff_pos[idx]
            dist     = np.sqrt(np.sum(diff_pos**2, axis=2))
            # Cycle through grav-only halo list
            for i,name in enumerate(f_dm['halos'].keys()):
                # Name of hydro halo
                bao_name = names_hydro[np.argmin(dist[i])]
                _ = f_dm['halos'][name].create_dataset('hydro_name', data=bao_name)
                # Distance of match
                _ = f_dm['halos'][name].create_dataset('match_dist', data=np.amin(dist[i]))
                good_match = 0
                if np.amin(dist[i]) < 2*f_dm['halos'][name]['Rvir'][()]:
                    good_match = 1
                _ = f_dm['halos'][name].create_dataset('good_match', data=good_match)
                # Properties of hydro halo
                for p in ['GPOS', 'M200c', 'M200m', 'M500c', 'M500m', 'Rvir']:
                    _ = f_dm['halos'][name].create_dataset('hydro_%s'%p, data=f_hydro['halos'][bao_name][p])


# add the 'has_profile' flag to matched halos
def add_has_profile(cat_dm):
    r"""Labels matched halos with "has_profile" flag

    Parameters
    ----------    
    cat_dm    =  DM matched catalog (.h5py file)
    """          
    
    bins = 10**np.arange(3, 5.4, .1)
    rng = np.random.default_rng(1328)
    with h5py.File(cat_dm, 'a') as cat_hdf5:
        names = np.array(list(cat_hdf5['halos'].keys()))
        m     = np.array([cat_hdf5['halos'][name]['M200c'][()] for name in cat_hdf5['halos'].keys()])
        good_match = np.array([cat_hdf5['halos'][name]['good_match'][()] for name in cat_hdf5['halos'].keys()])
        for name in names:
            try:
                del cat_hdf5['halos'][name]['has_profile']
            except:
                pass
            _ = cat_hdf5['halos'][name].create_dataset('has_profile', data=0)
            cat_hdf5['halos'][name]['has_profile'][()] = 0
        for bin_id in range(len(bins)-1):
            idx = ((m>=bins[bin_id])&(m<bins[bin_id+1])&(good_match==1)).nonzero()[0]
            if len(idx)>200:
                idx = rng.choice(idx, size=200, replace=False)
            for i in idx:
                cat_hdf5['halos'][names[i]]['has_profile'][()] = 1
                
                
# create .h5py catalog (DM + HYDRO matched halos)                
def create_catalog(catalog_file, sim_base_dm, sim_base_hydro, snap_num):
    r"""Create the final catalog to use for making mass maps

    Parameters
    ----------       
    cat_file       = h5py file for saving DM and matched HYDRO halos
    sim_base_dm    = DM original catalog (Magneticum)
    sim_base_hydro = HYDRO original catalog (Magneticum)
    snap           = snapshot number
    """
    
    # dump dm catalog
    print('Creating DM catalog')
    dump_catalog(sim_base_dm, snap_num, catalog_file)
    
    # dump hydro catalog
    print('Creating hydro catalog')
    catalog_file_hydro = 'cat_hydro_%s.h5py' %catalog_file[51:-5]
    dump_catalog(sim_base_hydro, snap_num, catalog_file_hydro)
    
    # match catalogs
    print('Matching DM and hydro halos')
    match_catalogs(catalog_file,catalog_file_hydro)

    # no needed anymore, hydro halos are saved in catalog_file along with the corresponding DM ones
    os.remove(catalog_file_hydro) 
       
    # assign "has_profile" flag 
    print('Assigning "has_profile" flag to matched halos')
    add_has_profile(catalog_file)
    
    print('Catalog created')
    
    
                
if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
