'''
Define the data types used by GalaxyClusters
'''

from collections import namedtuple

Position = namedtuple('Position', ['metadata', 'ra', 'dec'])
Profile = namedtuple('Profile', ['metadata', 'distances', 'dimension'])

for n in ['Position', 'Profile']:

    exec('%s00 = %s(*[None for i in %s._fields])'%(n, n, n))
    exec('%s_type = type(%s)'%(n, n))

