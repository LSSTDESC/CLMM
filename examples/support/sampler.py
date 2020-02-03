"""@file.py sampler.py
Functions for sampling (output either peak or full distribution)

"""

def sciopt(nfw_to_shear_profile, logm_0, args) :
    ''' Uses scipy optimize minimize to output the peak'''
    from scipy import optimize as spo

    return spo.minimize(nfw_to_shear_profile, logm_0,
                 args=[r1, gt_profile1, z1]).x



samplers = {
    'minimize':sciopt,
    
    }
