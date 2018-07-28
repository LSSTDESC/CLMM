"""
Define the custom data type
"""

import astropy
from astropy import table
import collections
from collections import namedtuple

"""
Data: A namedtuple tying values with units to the metadata of where the values came from and thus how the values are used

Parameters
----------
provenance: string, type of object that produced this data
    e.g. Profile.NFW
data: astropy table with column names and units

Notes
-----

"""

GCData = namedtuple('GCData', ['provenance', 'table'])
