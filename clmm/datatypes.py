'''
Define the data types used by GalaxyClusters
'''

from collections import namedtuple

Position = namedtuple('Position', ['metadata', 'ra', 'dec'])
Profile = namedtuple('Profile', ['metadata', 'distances', 'dimension'])

