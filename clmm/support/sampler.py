"""@file.py sampler.py
Functions for sampling (output either peak or full distribution)

"""


def sciopt(model_to_shear_profile, logm_0, args) :
    ''' Uses scipy optimize minimize to output the peak'''
    from scipy import optimize as spo

    return spo.minimize(model_to_shear_profile, logm_0,
                 args=args).x

def basinhopping(model_to_shear_profile, logm_0, args) :
    '''Uses basinhopping, a scipy global optimization function, to find the minimum '''
    from scipy import optimize as spo

    return spo.basinhopping(model_to_shear_profile, logm_0, minimizer_kwargs={'args':args}).x[0]


def scicurve_fit(profile_model,radius,profile,err_profile,**kwargs):
    '''Uses scipy.optimize.curve_fit to find best fit parameters'''
    
    from scipy import optimize as spo

    return spo.curve_fit(profile_model,
                    radius, profile,
                    sigma=err_profile, **kwargs)



samplers = {
    'minimize':sciopt,
    'basinhopping':basinhopping
    }

fitters = {
    'curve_fit':scicurve_fit,

    }