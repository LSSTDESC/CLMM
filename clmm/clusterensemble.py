"""@file clusterensemble.py
The Cluster Ensemble class
"""

from .galaxycluster import GalaxyCluster
from collections import Sequence

class ClusterEnsemble():
    """Object that contains a list of GalaxyCluster objects

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster ensemble
    gclist : list
        List of galaxy cluster objects
    """
    def __init__(self, unique_id, gclist):
        """Initializes a ClusterEnsemble object

        Parameters
        ----------
        unique_id : int or string
            Unique identifier of the galaxy cluster ensemble
        gclist : collections.Sequence
            Array-like Sequence of galaxy cluster objects

        Returns
        ---------

        """
        if isinstance(unique_id, (int, str)):
            unique_id = str(unique_id)
        else:
            raise TypeError('unique_id incorrect type: %s'%type(unique_id))
        if isinstance(gclist, Sequence):
            gclist = list(gclist)
        else:
            raise TypeError('gclist incorrect type: %s'%type(gclist))
        for gc in gclist:
            if ~isinstance(gc, GalaxyCluster):
                raise TypeError('gclist entry incorrect type: %s'%type(gc))

        self.unique_id = unique_id
        self.gclist = gclist

    def __getitem__(self, key):
        """Returns GalaxyCluster object at key in gclist"""
        if ~isinstance(key, int):
            raise TypeError('key incorrect type: %s'%type(key))
        
        return gclist[key]

    def __len__(self):
        """Returns length of ClusterEnsemble"""
        return len(gclist)

    def stack(self):
        """Produces a GalaxyCluster object by stacking elements of gclist

        Parameters
        ---------

        Returns
        ---------
        gc_stack : GalaxyCluster
            Stacked galaxy cluster generated from elements of self.gclist
        """
        return
