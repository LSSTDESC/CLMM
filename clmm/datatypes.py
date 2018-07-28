"""
Define the custom data type
"""

import astropy
from astropy import table
import collections
from collections import namedtuple

"""
GCData: A namedtuple tying values with units to the metadata of where the values came from and thus how the values are used

Parameters
----------
creator: what object (i.e. model, summarizer, inferrer)
specs: specifications of how the data was created
data: astropy table with column names and units

Notes
-----

"""
GCData = namedtuple('GCData', ['creator', 'specs', 'table'])
