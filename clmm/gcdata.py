"""
Define the custom data type
"""
from astropy.table import Table as APtable
import warnings
import pickle

from collections import OrderedDict


class GCMetaData(OrderedDict):
    r"""Object to store metadata, it always has a cosmo key with protective changes

    Attributes
    ----------
    protected: bool
        Protect cosmo key
    OrderedDict attributes
    """

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        if 'cosmo' not in self:
            self.__setitem__('cosmo', None, True)

    def __setitem__(self, item, value, force=False):
        if item == 'cosmo' and not force and \
            (self['cosmo'] is not None if 'cosmo' in self else False):
            raise ValueError('cosmo must be changed via update_cosmo or update_cosmo_ext_valid method')
        else:
            OrderedDict.__setitem__(self, item, value)
        return
    def __getitem__(self, item):
        """
        Make class accept all letter casings
        """
        if isinstance(item, str):
            item = {n.lower():n for n in self.keys()}[item.lower()]
        out = OrderedDict.__getitem__(self, item)
        return out


class GCData(APtable):
    """
    GCData: A data objetc for gcdata. Right now it behaves as an astropy table,
    with the following modifications: `__getitem__` is case independent;
    The attribute .meta['cosmo'] is protected and can only be changed via
    update_cosmo or update_cosmo_ext_valid methods;

    Parameters
    ----------
    meta: dict
        Dictionary with metadata for this object

    Same as astropy tables
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs: Same used for astropy tables
        """
        APtable.__init__(self, *args, **kwargs)
        metakwargs = kwargs['meta'] if 'meta' in kwargs else {}
        metawkargs = {} if metakwargs is None else metakwargs
        self.meta = GCMetaData(**metakwargs)

    def __repr__(self):
        """Generates string for repr(GCData)"""
        output = f'{self.__class__.__name__}('
        output+= ', '.join([f'{key}={value!r}'
            for key, value in self.meta.items()]
                +['columns: '+', '.join(self.colnames)])
        output+= ')'
        return output

    def __str__(self):
        """Generates string for print(GCData)"""
        output = f'{self.__class__.__name__}\n> defined by:'
        output+= ', '.join([f'{key}={str(value)}'
            for key, value in self.meta.items()])
        output += f'\n> with columns: '
        output += ', '.join(self.colnames)
        output += f'\n> and {len(self)} objects'
        output += '\n'+APtable.__str__(self)
        return output

    def _repr_html_(self):
        """Generates string for display(GCData)"""
        output = f'<b>{self.__class__.__name__}<br> defined by:</b> '
        output+= ', '.join([f'{key}={str(value)}'
            for key, value in self.meta.items()])
        output += f'<br><b>with columns:</b> '
        output += ', '.join(self.colnames)
        output += f'<br>and {len(self)} objects'
        table = APtable._repr_html_(self)
        return output+'<br>'+'</i>'.join(table.split('</i>')[1:])

    def __getitem__(self, item):
        """
        Makes sure GCData keeps its properties after [] operations are used.
        It also makes all letter casings accepted

        Returns
        -------
        GCData
            Data with [] operations applied
        """
        if isinstance(item, str):
            name_dict = {n.lower():n for n in self.colnames}
            item = item.lower()
            item = ','.join([name_dict[i] for i in item.split(',')])
        out = APtable.__getitem__(self, item)
        return out

    def update_cosmo_ext_valid(self, gcdata, cosmo, overwrite=False):
        r"""Updates cosmo metadata if the same as in gcdata

        Parameters
        ----------
        gcdata: GCData
            Table to check if same cosmology
        cosmo: clmm.Cosmology
            Cosmology
        overwrite: bool
            Overwrites the current cosmo metadata. If false raises Error when cosmologies are different.

        Returns
        -------
        None
        """
        cosmo_desc = cosmo.get_desc() if cosmo else None
        if cosmo_desc:
            cosmo_gcdata = gcdata.meta['cosmo']
            if cosmo_gcdata and cosmo_gcdata != cosmo_desc:
                if overwrite:
                    warnings.warn(f'input cosmo ({cosmo_desc}) overwriting gcdata cosmo ({cosmo_gcdata})')
                else:
                    raise TypeError(f'input cosmo ({cosmo_desc}) differs from gcdata cosmo ({cosmo_gcdata})')
            self.meta.__setitem__('cosmo', cosmo_desc, force=True)
        return

    def update_cosmo(self, cosmo, overwrite=False):
        r"""Updates cosmo metadata if not present

        Parameters
        ----------
        cosmo: clmm.Cosmology
            Cosmology
        overwrite: bool
            Overwrites the current cosmo metadata. If false raises Error when cosmologies are different.

        Returns
        -------
        None
        """
        self.update_cosmo_ext_valid(self, cosmo, overwrite=overwrite)
        return

"""
Additional functions specific to clmm.GCData
Note: Not being used anymore
"""
#
#def confirm_GCData(data):
#    """
#    Typechecks GCData elements
#    """
#    # if not (type(data.creator) == str and type(data.specs) == dict and type(data.values) == astropy.Table):
#    #     raise TypeError('GCData creator should be string, specs should be dict, values should be astropy table')
#    pass
#
#
#def read_GCData(filename):
#    """
#    Reads GCData from file
#    """
#    pass
#
#
#def write_GCData(data, filename):
#    """
#    Writes GCData to file
#    """
#    pass
#
#
#def check_subdict(lookup_dict, reference_dict):
#    '''
#    Checks if a dictionary is a subset of another dictionary, with the same keys and values
#
#    Parameters
#    ----------
#    lookup_dict: dict
#        Subset dictionary
#    reference_dict: dict
#        Main dictionary
#
#    Returns
#    -------
#    bool
#        If a lookup_dict is a subset of reference_dict
#    '''
#    return lookup_dict.items() <= reference_dict.items()
#
#
#def find_in_datalist(lookup_specs, datalist, exact=False):
#    '''
#    Finds data with given specs in a datalist, allows for partial match
#
#    Parameters
#    ----------
#    datalist: list
#        List of clmm.GCData objects to search for lookup_specs
#    lookup_specs: dict
#        Specs required
#    exact: boolean, optional
#        Does the match have to be symmetric?
#
#    Returns
#    -------
#    found: list
#        List of clmm.GCData object data with required creator and set of specs
#    '''
#    found = []
#    for data in datalist:
#        if check_subdict(lookup_specs, data.specs) :
#            found.append(data)
#    if exact:
#        for match in found:
#            if check_subdict(match.specs, lookup_specs):
#                return [match]
#        found = []
#    return found
