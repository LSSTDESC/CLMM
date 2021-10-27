"""@file clusterensemble.py
The Cluster Ensemble class
"""

from .gcdata import GCData
from .galaxycluster import GalaxyCluster
from collections import Sequence

class ClusterEnsemble():
    """Object that contains a list of GalaxyCluster objects

    Attributes
    ----------
    unique_id : int or string
        Unique identifier of the galaxy cluster ensemble
    data : GCData
        Table with galaxy clusters data (i. e. ids, profiles, redshifts).
    id_dict: dict
        Dictionary of indicies given the cluster id
    """
    def __init__(self, unique_id, *args, **kwargs):
        """Initializes a ClusterEnsemble object

        Parameters
        ----------
        unique_id : int or string
            Unique identifier of the galaxy cluster ensemble
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
        self.data = GCData
        self.id_dict = {}
        if len(args)>0 or len(kwargs)>0:
            self._add_values(*args, **kwargs)

    def _add_values(self, gc_list, gc_cols):
        """Add values for all attributes

        Parameters
        ----------
        gc_list : list, tuple
            List of GalaxyCluster objects.
        gc_cols : list, tuple
            List of GalaxyCluster objects.
        """
        for gc in gc_list:
            self.add_cl_profile(gc)
        self.id_dict = {i:ind for ind, i in enumerate(self['id'])}

    def __getitem__(self, item):
        """Returns self.data[item]"""
        return self.data[item]

    def __len__(self):
        """Returns length of ClusterEnsemble"""
        return len(self.data)

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
