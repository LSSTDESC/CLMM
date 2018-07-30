'''
GalaxyCluster is the fundamental object in clmm
'''
from datatypes import *

class GalaxyCluster():
    '''
    Object that contains the information associated with a galaxy cluster
    '''

    def __init__(self, initial_data=None, homelocal='.'):
        '''
        Parameters
        ----------
        initial_data: list, clmm.GCData objects, optional
            Initial data to associate with GalaxyCluster object
        homelocal: string, optionnal
            Path to save cluster properties
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
            Creator that will be searched in GalaxyCluster object

        Returns
        -------
        bool
            Existence of the test_creator in GalaxyCluster object
        '''

        return test_creator in self.data


    def _find(self, lookup_creator, lookup_specs, exact=False):
        '''
        Finds data with a specific creator and specs in GalaxyCluster object
        allows for partial match

        Parameters
        ----------
        creator: string
            Creator that will be searched in GalaxyCluster object
        specs: dict
            Specs requiered inside the creator
        exact: boolean
            Does it have to be a symmetric match?

        Returns
        -------
        list, None
            List of clmm.GCData object data with required creator and set of specs
            if no objects are found, returns None
        '''

        if self._check_creator(lookup_creator):
            return find_in_datalist(lookup_specs, self.data[lookup_creator])
        else:
            return None


    def _find_exact(self, lookup_creator, lookup_specs):
        '''
        Finds data with a specific cretor and specs in GalaxyCluster object
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
            if not found, returns None

        '''

        if self._check_creator(lookup_creator):
            return Aux.find_in_datalist_exact(self.data[lookup_creator], lookup_specs)
        else:
            return None


    def _check_datatype(self, incoming_data):
        return type(incoming_data) == GCData


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
         '''

        if not self._check_datatype(incoming_data):
            if self._debugprint: print('wrong type')
            return

        if self._check_creator(incoming_data.creator):
            found_data = Aux.find_in_datalist_exact(self.data[incoming_data.creator], incoming_data.specs)

            if found_data:
                if replace:
                    print('Overwriting this data:')
                    Aux.remove_in_datalist(self.data[incoming_data.creator], incoming_data.specs)
                    self.data[incoming_data.creator].append( incoming_data )

                else:
                    print('Data with this creator & specs already exists. Add replace=True key to replace it.')
                    print('Current:')

                print('\t', found_data)

            else:
                self.data[incoming_data.creator].append( incoming_data )

        else:
            self.data[incoming_data.creator] = [ incoming_data ]


    def _remove(self, creator, specs):

        if self._check_creator(creator):
            if Aux.find_in_datalist_exact(self.data[creator], specs):
                Aux.remove_in_datalist(self.data[creator], specs)
        else:
            print('*** ERROR *** - creator "%s" not found in datalist'%creator)
