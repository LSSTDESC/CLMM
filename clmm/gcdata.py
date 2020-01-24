"""
Define the custom data type
"""

# from astropy import table
from collections import namedtuple
import pickle


GCData_ = namedtuple('GCData', ['creator', 'specs', 'values'])


class GCData(GCData_):
    """
    GCData: A namedtuple tying values with units to the metadata of where the values came from
    and thus how the values are used

    Parameters
    ----------
    creator: string
        Super class of the object that made the data.
    specs: dict
        Specifications of how the data was created, what are the properties of the data.
        If the data was created by a function, what inputs were used.
    values: astropy.Table
        Data with units
    """
    pass


"""
Additional functions specific to clmm.GCData
"""

def confirm_GCData(data):
    """
    Typechecks GCData elements
    """
    # if not (type(data.creator) == str and type(data.specs) == dict and type(data.values) == astropy.Table):
    #     raise TypeError('GCData creator should be string, specs should be dict, values should be astropy table')
    pass


def read_GCData(filename):
    """
    Reads GCData from file
    """
    pass


def write_GCData(data, filename):
    """
    Writes GCData to file
    """
    pass


def check_subdict(lookup_dict, reference_dict):
    '''
    Checks if a dictionary is a subset of another dictionary, with the same keys and values

    Parameters
    ----------
    lookup_dict: dict
        Subset dictionary
    reference_dict: dict
        Main dictionary

    Returns
    -------
    bool
        If a lookup_dict is a subset of reference_dict
    '''
    return lookup_dict.items() <= reference_dict.items()


def find_in_datalist(lookup_specs, datalist, exact=False):
    '''
    Finds data with given specs in a datalist, allows for partial match

    Parameters
    ----------
    datalist: list
        List of clmm.GCData objects to search for lookup_specs
    lookup_specs: dict
        Specs required
    exact: boolean, optional
        Does the match have to be symmetric?

    Returns
    -------
    found: list
        List of clmm.GCData object data with required creator and set of specs
    '''
    found = []
    for data in datalist:
        if check_subdict(lookup_specs, data.specs) :
            found.append(data)
    if exact:
        for match in found:
            if check_subdict(match.specs, lookup_specs):
                return [match]
        found = [] 
    return found
