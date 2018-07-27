'''
Define CalaxyClusters class
Who to blame for problems: Michel A.
'''

Position = namedtuple('Position', ['ra', 'dec'])
Profile = namedtuple('Profile', ['distances', 'dimension'])

for n in ['Position', 'Pofile']:

    exec('%00 = %s(*[None for i in %s._fields])'%(n, n))
    exec('%s_type = type(%)'%(n, n))


class GalaxyCluster():
    '''
    - read data
    - write data
    - disting data
    '''

    def __init__(self):
        '''
        '''
        self.data = {}
        self.homelocal = '.'

    def _add(self, added_data):
        '''
        added_data - dict with all the data in the correct formats
        '''

        for name, dat in added_data.items():

            for d in self.data:

                if type(d) == type(dat):

                    if d.metadata == dat.metadata:

                        print('Overwritting %s data')

            self.data[name] = dat

    def _listprofiles():
        '''
        '''
        print([d.dimension for d in self.data['profiles']])
        check_data()

    def showdata(self):
        '''
        '''

        print(self.data.keys())

    def measure_profile(self):
        '''
        '''

    def fit_profile(self):
        '''
        '''

    def func(self):
        '''
        '''
