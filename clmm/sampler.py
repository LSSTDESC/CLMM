"""@file.py sampler.py
Functions for sampling (output either peak or full distribution)

"""

def sciopt() :
    ''' Uses scipy optimize minimize to output the peak'''
    from scipy import optimize as spo

    # spo.minimize(nfw_to_shear_profile, logm_0,
    #                     args=[r1, gt_profile1, z1]).x

    pass



samplers = {
    'minimize':sciopt,
    
    }
