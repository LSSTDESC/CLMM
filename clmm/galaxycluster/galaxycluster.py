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
        if test_creator in self.data:
            return self.data
        else:
            return False

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
        if lookup_creator in self.data:
            return find_in_datalist(lookup_specs, self.data[lookup_creator], exact=exact)
        else:
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
                # is the creator already there?
                #  false: make it
                #  true: are specs already there?
                #   false: append it
                #   true: do we want to overwrite?
                #    false: exit
                #    true: remove, then add
         '''
        if not type(incoming_data) == GCData:
            # change to raise/except
            if self._debugprint: print('wrong type')
            return
        if not incoming_data.creator in self.data:
            self.data[incoming_data.creator] = [incoming_data]
        else:
            found_data = find_in_dataset(incoming_data.specs, self.data[incoming_data.creator], exact=True)
            if not found_data
                self.data[incoming_data.creator].append(incoming_data)
            else:
                if not replace:
                    print('Data with this creator & specs already exists. Add replace=True key to replace it.')
                    print('Current:', found_data))
                else:
                    print('Overwriting this data:')
                    self.data[incoming_data.creator].remove(found_data)
                    self.data[incoming_data.creator].append(incoming_data)
        return

    def _remove(self, incoming_data):
        """
        Removes data from GalaxyCluster

        Parameters
        ----------
        incoming_data: GCData object
            the data to be removed

        Notes
        -----
        """
        if incoming_data.creator in self.data:
            if not find_in_dataset(incoming_data.specs, self.data[incoming_data.creator], exact=True):
                # change to raise/except
                print('*** specs ERROR *** - incoming data not found in datalist')
            else:
                self.data[incoming_data.creator].remove(incoming_data.specs)
        else:
            print('*** creator ERROR *** - incoming data not found in datalist')
        return
