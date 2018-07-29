'''
Define CalaxyClusters class
Who to blame for problems: Michel A.
'''

from datatypes import GCData_type

class GalaxyCluster():
    '''
    Object that contains the properties of a galaxy cluster.
    '''

    def __init__(self, initial_data = None, homelocal = '.', datatype = GCData_type):
        '''
        Parameters
        ----------
        initial_data: list of clmm.GCData objects, optional
            initial data to associate with GalaxyCluster object
        homelocal: string, optionnal
            path to save cluster properties

        Notes
        -----

        '''
        self.data = {}
        self.homelocal = homelocal
        self.datatype = datatype

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
        list, None
            list of clmm.GCData object data with required creator and set of specs
            if not objects are found, returns None
        '''

        if self._check_creator(lookup_creator):

            return AuxFuncs._find_in_datalist(self.data[lookup_creator], lookup_specs)

        else:

            return None
            
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
            if not found, returns None

        '''

        if self._check_creator(lookup_creator):

            return AuxFuncs._find_in_datalist_exact(self.data[lookup_creator], lookup_specs)

        else:

            return None

    def _check_datatype(self, incoming_data):

        if type(incoming_data) != self.datatype:

            if self._debugprint:

                print('*** ERROR *** - incoming_data is type "%s", must be "%s"'
                            %(type(incoming_data), self.datatype) )

            return False

        return True

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

        if not self._check_datatype(incoming_data):

            return

        if self._check_creator(incoming_data.creator):

            found_data = AuxFuncs._find_in_datalist_exact(self.data[incoming_data.creator], incoming_data.specs)

            if found_data:

                if replace:

                    print('Overwritting this data:')
                    AuxFuncs._remove_in_datalist(self.data[incoming_data.creator], incoming_data.specs)
                    self.data[incoming_data.creator].append( incoming_data )

                else:

                    print('Data with this creator & specs already exists. Add replace=True key to replace it.')
                    print('Current:')

                print('   ', found_data)

            else:

                self.data[incoming_data.creator].append( incoming_data )

        else:

            self.data[incoming_data.creator] = [ incoming_data ]

    def _remove(self, creator, specs):

        if self._check_creator(creator):

            if AuxFuncs._find_in_datalist_exact(self.data[creator], specs):

                AuxFuncs._remove_in_datalist(self.data[creator], specs)

        else:

            print('*** ERROR *** - creator "%s" not found in datalist'%creator)


class AuxFuncs():

    def _check_subdict(test_dict, main_dict):
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

        for test_key, test_value in test_dict.items():

            if test_key in main_dict:

                if test_value != main_dict[test_key]:

                        return False

            else:

                return False

        return True

    def _find_in_datalist(datalist, lookup_specs):
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
        list, None
            list of clmm.GCData object data with required creator and set of specs
            if not objects are found, returns None
        '''

        found = []

        for data in datalist:

            if AuxFuncs._check_subdict(lookup_specs, data.specs) :

                found.append( data )

        if len(found) == 0:

            print('*** WARNING *** no data found with these specification!')
            found = None

        elif len(found) > 1:

            print('*** WARNING *** multiple data found with these specification!')
            
        return found

    def _find_ind_in_datalist_exact(datalist, lookup_specs):
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
        int, None
            index of clmm.GCData object in datalist with required creator and set of specs
            if not found, returns None

        '''

        for ind, data in enumerate(datalist):

            if lookup_specs == data.specs :

                return ind

        else:

            print('*** ERROR *** - no data found with these specification!')
            return None
    
    def _find_in_datalist_exact(datalist, lookup_specs):
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
            if not found, returns None

        '''

        for data in datalist:

            if lookup_specs == data.specs :

                return data

        else:

            print('*** ERROR *** - no data found with these specification!')
            return None

    def _remove_in_datalist(datalist, specs):

        ind = AuxFuncs._find_ind_in_datalist_exact(datalist, specs)

        if ind is not None:

            del datalist[ind]

