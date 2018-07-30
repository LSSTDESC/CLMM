'''
Auxiliary functions used by GalaxyCluster object
'''

def check_subdict(test_dict, main_dict):
    '''
    Checks if a dictionary is a subset of another dictionary, with the same keys and values

    Parameters
    ----------
    test_dict: dict
        Subset dictionary
    main_dict: dict
        Main dictionary

    Returns
    -------
    bool
        If a test_dict is a subset of main_dict
    '''
    return test_dict.items() <= main_dict.items()


def find_in_datalist(datalist, lookup_specs):
    '''
    Finds data with given specs in a datalist, allows for partial match

    Parameters
    ----------
    datalist: list
        List of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        Specs required

    Returns
    -------
    list
        List of clmm.GCData object data with required creator and set of specs
    None
        If no objects are found
    '''

    found = []

    for data in datalist:
        if check_subdict(lookup_specs, data.specs) :
            found.append(data)

    if len(found) == 0:
        print('*** WARNING *** no data found with these specification!')
        found = None

    elif len(found) > 1:
        print('*** WARNING *** multiple data found with these specification!')
        
    return found


def find_ind_in_datalist_exact(datalist, lookup_specs):
    '''
    Finds data with given specs in a datalist,
    requires exact match

    Parameters
    ----------
    datalist: list
        List of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        Specs required

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


def find_in_datalist_exact(datalist, lookup_specs):
    '''
    Finds data with given specs in a datalist, requires exact match

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


def remove_in_datalist(datalist, specs):
    ind = find_ind_in_datalist_exact(datalist, specs)
    if ind is not None:
        del datalist[ind]

