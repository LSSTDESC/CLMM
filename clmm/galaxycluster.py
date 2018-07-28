'''
Define CalaxyClusters class
Who to blame for problems: Michel A.
'''

from datatypes import *

class GalaxyCluster():
    '''
    Object that contains the properties of a galaxy cluster.
    It must have this opperations:
        - read data
        - write data
        - disting data
    '''

    def __init__(self):
        '''
        '''
        self.data = {}
        self.homelocal = '.'

    def _add(self, data_external):
        '''
        data_external - dict with all the data in the correct formats
        '''

        for name, data_ext in data_external.items():

            for data_int in self.data.values():

                if type(data_int) == type(data_ext):

                    if data_int.metadata == data_ext.metadata:

                        print('Overwritting %s[%s] data'%(name, data_ext.metadata))

            self.data[name] = data_ext

    def _listprofiles():
        '''
        '''
        print([d.dimension for d in self.data['profiles']])
        check_data()

    def showdata(self):
        '''
        '''

        for n, d in self.data.values():

            print(n, d.metadata)

    def measure_profile(self):
        '''
        '''

    def fit_profile(self):
        '''
        '''

    def func(self):
        '''
        '''
