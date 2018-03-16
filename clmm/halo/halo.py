'''Here we populate the different attributes and methods of a halo.'''

from profile import Profile

def Halo :
    '''
    
    config_params : read in from config file, all of which make this Halo instance unique

    data : gets used in single inference, might need to be calculated from read-in data, e.g. a Profile object, so self.profile.

    '''
    
    def __init__(self, config_file) :
        
        pass

    def get_profile(self) :
        '''Instantiate a Profile from items in the config file'''

        self.profile = Profile(config_file, **kw)

        pass
    


halo_101 = Halo(config_file)

halo_101.profile
        

