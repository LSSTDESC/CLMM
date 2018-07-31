'''
GalaxyCluster is the fundamental object in clmm
'''

import pickle

from clmm.core import datatypes 
from clmm.core.datatypes import GCData, find_in_datalist

class GalaxyCluster():
    '''
    Object that contains the information associated with a galaxy cluster

    Attributes
    ----------
    data: dictionary
        Dictionary with creators as keys and a list of clmm.GCData objects as values
    homelocal: string
        Path to save cluster properties
    '''

    def __init__(self, initial_data=None, homelocal='.'):
        '''
        Parameters
        ----------
        initial_data: list, clmm.GCData objects, optional
            Initial data to associate with GalaxyCluster object
        homelocal: string, optional
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
        lookup_creator: string
            Creator that will be searched in GalaxyCluster object
        lookup_specs: dict
            Specs requiered inside the creator
        exact: boolean
            Does it have to be a symmetric match?

        Returns
        -------
        found: list, clmm.GCData objects or boolean
            List of clmm.GCData object data with required creator and set of specs
            if no objects are found, returns False
        '''
        if lookup_creator in self.data:
            found = find_in_datalist(lookup_specs, self.data[lookup_creator], exact=exact)
        else:
            found = False
        return(found)

    def add_data(self, incoming_data, force=False):
        '''
        Parameters
        ----------
        incoming_data: clmm.GCData object
            new data to associate with GalaxyCluster object
        force: bool, optional
            replace in the case of data with same creator, specs already exists

        Notes
        -----
        This function asks, "is the creator already there?" if not, make it.
        If it is, are specs already there? If not, append it.
        If they are, do we want to overwrite? If no, exit. If so replace.
        '''
        if not type(incoming_data) == GCData:
            raise TypeError('incoming data of wrong type')
        if not incoming_data.creator in self.data:
            self.data[incoming_data.creator] = [incoming_data]
        else:
            found_data = find_in_datalist(incoming_data.specs, self.data[incoming_data.creator], exact=True)
            if found_data == []:
                self.data[incoming_data.creator].append(incoming_data)
            else:
                if not force:
                    raise ValueError('Data with this creator & specs already exists. Add force=True keyword to replace it.')
                else:
                    print(found_data)
                    self.data[incoming_data.creator].remove(found_data[0])
                    self.data[incoming_data.creator].append(incoming_data)
        return

    def remove_data(self, incoming_creator, incoming_specs):
        """
        Removes data from GalaxyCluster

        Parameters
        ----------
        incoming_data: GCData object
            the data to be removed
        """
        if incoming_creator in self.data:
            exact_data = find_in_datalist(incoming_specs, self.data[incoming_creator], exact=True)
            if exact_data != []:
                self.data[incoming_creator].remove(exact_data[0])
                return
        raise ValueError('incoming data not found in GalaxyCluster')

    def _setup_dir(self, homelocal):
        """
        Checks if directory exists, otherwise makes it
        """
        raise ValueError('This function is currently empty. Sorry for the inconvenience.')

    def read_GC(self, filename, lookup_creators=None, lookup_specs=None):
        """
        Reads in a pickled GalaxyCluster's data from saved versions
        """
        raise ValueError('This function is currently empty. Sorry for the inconvenience.')

    def write_GC(self, filename, lookup_creator=None, lookup_specs=None):
        """
        Pickles GalaxyCluster's data and saves it
        """
        raise ValueError('This function is currently empty. Sorry for the inconvenience.')
