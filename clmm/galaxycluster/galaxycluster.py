'''
GalaxyCluster is the fundamental object in clmm
'''
import cPickle
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

        if initial_data is not None:
            self.add_data(initial_data)

    def find_data(self, lookup_creator, lookup_specs, exact=False):
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

    def add_data(self, incoming_data, force=False):
        '''
        Parameters
        ----------
        incoming_data: clmm.GCData object
            new data to associate with GalaxyCluster object
        incoming_metadata: dict
            new metadata for GalaxyCluster to use to distinguish from other clmm.GCData with same provenance
        force: bool, optional
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
            raise TypeError('incoming data of wrong type')
        if not incoming_data.creator in self.data:
            self.data[incoming_data.creator] = [incoming_data]
        else:
            found_data = find_in_dataset(incoming_data.specs, self.data[incoming_data.creator], exact=True)
            if not found_data
                self.data[incoming_data.creator].append(incoming_data)
            else:
                if not force:
                    raise ValueError('Data with this creator & specs already exists. Add force=True keyword to replace it.')
                else:
                    self.data[incoming_data.creator].remove(found_data)
                    self.data[incoming_data.creator].append(incoming_data)
        return

    def remove_data(self, incoming_data):
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
            if find_in_dataset(incoming_data.specs, self.data[incoming_data.creator], exact=True):
                self.data[incoming_data.creator].remove(incoming_data.specs)
                return
        raise ValueError('incoming data not found in GalaxyCluster')

    def load_GC():
        pass

    def save_GC():
        pass
