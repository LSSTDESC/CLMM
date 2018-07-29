'''
Define CalaxyClusters class
Who to blame for problems: Michel A.
'''


class GalaxyCluster():
    '''
    Object that contains the properties of a galaxy cluster.

    Parameters
    ----------

        data: dictionary with all cluster properties.
            Each entry  must contain
        homelocal: path to save cluster properties


    It must have this opperations:
        - read data
        - write data
        - disting data
    '''

    def __init__(self, initial_data=None, homelocal='.'):
        '''
        Notes
        -----

        '''
        self.data = []
        self.homelocal = homelocal
        self.creators = {}

        if initial_data is not None:
            self._add(initial_data)

    def _add(self, incoming_data, replace=False):
        '''
        Parameters
        ----------
        incoming_data: clmm.GCData object
            new data to associate with GalaxyCluster object
        incoming_metadata: dict
            new metadata for GalaxyCluster to use to distinguish from other clmm.GCData with same provenance

        Notes
        -----

         '''

        if incoming_metadata.creator in self.creators:

            self.creators[incoming_data.creator].append( len(self.data) )

            for i in self.creators[incoming_metadata.creator]:

                if incoming_metadata.specs = self.data[i].specs:

                    if not replace:

                        print('Data "%s" already exists, to replace it, add replace=True key.')
                        print('Current:')
                        self._show(metadata)
                        return 0

                    else:

                        print('Overwritting "%s" data:'%(name))
                        self._show(metadata)

        else:

            self.creators[incoming_data.creator] = [ len(self.data) ]

        self.data.append( incomming_data )

    def _find(self, input_creator, input_metadata):
        '''
        still in construction, must figure out what to do if self.data has more than one data
            with the required creator and specs
        '''

        right_i = None

        if input_creator in self.creators:

            for i in self.creators[input_creator]:

                right_i  = i

                for key, value in input_metadata:

                    if key in self.data[i].specs:

                        if value != self.data[i].specs[key]

                            right_i = None
                    else:

                        print('ERROR, key "%s" does not exist in %s'%(key, input_creator))
                        return 0

            return self.data[right_i]

    def _listprofiles():
        '''
        '''
        print([d.dimension for d in self.data['profiles']])
        check_data()

    def showdata(self):
        '''
        '''

        for n, d in self.data.values():

            print(n, d.provenance)

    def measure_profile(self):
        '''
        '''

    def fit_profile(self):
        '''
        '''

    def func(self):
        '''
        '''
