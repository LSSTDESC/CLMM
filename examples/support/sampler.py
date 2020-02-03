"""@file.py sampler.py
Functions for sampling (output either peak or full distribution)

"""

def sciopt(model_to_shear_profile, logm_0, args) :
    ''' Uses scipy optimize minimize to output the peak'''
    from scipy import optimize as spo

    return spo.minimize(model_to_shear_profile, logm_0,
                 args=args).x



samplers = {
    'minimize':sciopt,
    
    }
