'''
Define CalaxyClusters class
Who to blame for problems: Michel A.
'''


class GalaxyCluster():
    '''
    Object that contains the properties of a galaxy cluster.
    '''

    def __init__(self, initial_data=None, homelocal='.'):
        '''
        Parameters
        ----------
        data: list of clmm.GCData objects, optional
            initial data to associate with GalaxyCluster object
        homelocal: string, optionnal
            path to save cluster properties

        Notes
        -----

        '''
        self.data = {}
        self.homelocal = homelocal
        self._debugprint = False

        if initial_data is not None:
            self._add(initial_data)
    
    def _check_creator(self, test_creator):
        '''
        Checks if a creator exists in GalaxyCluster object

        Parameters
        ----------
        test_creator: string
            creator that will be searched in GalaxyCluster object

        Returns
        -------
        bool
            existence of the test_creator in GalaxyCluster object
        '''

        return test_creator in self.data

    def _check_subdict(self, test_dict, main_dict):
        '''
        Checks if a dictionary is a subset of another dictionary, with the same keys and values

        Parameters
        ----------
        test_dict: dict
            subset dictionary
        main_dict: dict
            main dictionary

        Returns
        -------
        bool
            if a test_dict is a subset of main_dict
        '''

        for test_key, test_value in test_dict:

            if test_key in main_dict:

                if test_value != main_dict[test_key]:

                        return False

            else:

                return False

        return True

    def _find_in_datalist(self, datalist, lookup_specs):
        '''
        Finds data with given specs in a datalist,
            allows for partial match

        Parameters
        ----------
        datalist: list
            list of clmm.GCData objects to search for lookup_specs
        lookup_specs: dict
            specs required

        Returns
        -------
        list, False
            list of clmm.GCData object data with required creator and set of specs
            if not objects are found, returns False
        '''

        found = []

        for data in datalist:

            if self._check_subdict(lookup_specs, data.specs) :

                found.append( data )

        if len(found) == 0:

            print('WARNING, no data found with these specification!')
            found = False

        if len(found) > 1:

            print('WARNING, multiple data found with these specification!')
            
        return found
    
    def _find_in_datalist_exact(self, datalist, lookup_specs):
        '''
        Finds data with given specs in a datalist,
            requires exact match

        Parameters
        ----------
        datalist: list
            list of clmm.GCData objects to search for lookup_specs
        lookup_specs: dict
            specs required

        Returns
        -------
        list
            list of clmm.GCData object data with required creator and set of specs
            if not found, returns False

        '''

        for data in self.data[lookup_creator]:

            if lookup_specs == data.specs :

                return data

        else:

            print('ERROR, no data found with these specification!')
            return False

    def _find(self, lookup_creator, lookup_specs):
        '''
        Finds data with a specific cretor and specs in GalaxyCluster object
            allows for partial match

        Parameters
        ----------
        creator: string
            creator that will be searched in GalaxyCluster object
        specs: dict
            specs requiered inside the creator

        Returns
        -------
        list, False
            list of clmm.GCData object data with required creator and set of specs
            if not objects are found, returns False
        '''

        if self._check_creator(lookup_creator):

            for data in self.data[lookup_creator]:

                if self._check_subdict(lookup_specs, data.specs) :

                    found.append( data )

        if len(found) == 0:

            print('WARNING, no data found with these specification!')
            found = False

        if len(found) > 1:

            print('WARNING, multiple data found with these specification!')
            
        return found
    
    def _find_exact(self, lookup_creator, lookup_specs):
        '''
        Finds data with a specsific cretor and specs in GalaxyCluster object
            requires exact match

        Parameters
        ----------
        creator: string
            creator that will be searched in GalaxyCluster object
        specs: dict
            specs requiered inside the creator

        Returns
        -------
        list
            list of clmm.GCData object data with required creator and set of specs
            if not found, returns False

        '''

        if self._check_creator(lookup_creator):

            for data in self.data[lookup_creator]:

                if lookup_specs == data.specs :

                    return data

        else:

            print('ERROR, no data found with these specification!')
            return False

    def _add(self, incoming_data, replace=False):
        '''
        Parameters
        ----------
        incoming_data: clmm.GCData object
            new data to associate with GalaxyCluster object
        incoming_metadata: dict
            new metadata for GalaxyCluster to use to distinguish from other clmm.GCData with same provenance
        replace: bool, optional
            replace in the case of data with same creator, specs already exists
       
        Notes
        -----

         '''

        if self._check_creator(incoming_data.creator):

            if self._check_specs_exact(incoming_data.creator, incoming_data.specs):

                print('Data with this creator, specs already exists, to replace it, add replace=True key.')
                print('Current:')
                self._show(metadata)
                        return 0

                    else:

                        print('Overwritting "%s" data:'%(name))
                        self._show(metadata)

        else:

            self.data[incoming_data.creator].append( incomming_data )


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
