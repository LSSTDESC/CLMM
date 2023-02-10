"""
Define the custom data type
"""
import warnings
from collections import OrderedDict
from astropy.table import Table as APtable
import numpy as np


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
            raise ValueError(
                'cosmo must be changed via update_cosmo or update_cosmo_ext_valid method')
        OrderedDict.__setitem__(self, item, value)

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
    GCData: A data objetc for gcdata. Right now it behaves as an astropy table, with the following
    modifications: `__getitem__` is case independent;
    The attribute .meta['cosmo'] is protected and
    can only be changed via update_cosmo or update_cosmo_ext_valid methods;

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
        metakwargs = {} if metakwargs is None else metakwargs
        self.meta = GCMetaData(**metakwargs)
        # this attribute is set when source galaxies have p(z)
        self.pzpdf_info = {'type': None}

    def _str_colnames(self):
        """Colnames in comma separated str"""
        return ', '.join(self.colnames)

    def _str_meta_(self):
        """Metadata in comma separated str"""
        return ', '.join([f'{key}={value!r}'
            for key, value in self.meta.items()])

    def __repr__(self):
        """Generates string for repr(GCData)"""
        description = [self._str_meta_(), 'columns: '+self._str_colnames()]
        return f'{self.__class__.__name__}({", ".join(description)})'

    def __str__(self):
        """Generates string for print(GCData)"""
        return (
            f'{self.__class__.__name__}'
            f'\n> defined by: {self._str_meta_()}'
            f'\n> with columns: {self._str_colnames()}'
            f'\n> {len(self)} objects'
            f'\n{APtable.__str__(self)}'
            )

    def _html_table(self):
        """Get html table for display"""
        return '</i>'.join(APtable._repr_html_(self).split('</i>')[1:])

    def _repr_html_(self):
        """Generates string for display(GCData)"""
        return (
            f'<b>{self.__class__.__name__}</b>'
            f'<br> <b>defined by:</b> {self._str_meta_()}'
            f'<br> <b>with columns:</b> {self._str_colnames()}'
            f'<br> {len(self)} objects'
            f'<br> {self._html_table()}'
            )

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
        # sub cols or sub rows
        if not isinstance(item, (str, int, np.int64)):
            out.pzpdf_info = self.pzpdf_info
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
            Overwrites the current cosmo metadata. If false raises Error when cosmologies are
            different.

        Returns
        -------
        None
        """
        cosmo_desc = cosmo.get_desc() if cosmo else None
        if cosmo_desc:
            cosmo_gcdata = gcdata.meta['cosmo']
            if cosmo_gcdata and cosmo_gcdata != cosmo_desc:
                if overwrite:
                    warnings.warn(
                        f'input cosmo ({cosmo_desc}) overwriting gcdata cosmo ({cosmo_gcdata})')
                else:
                    raise TypeError(
                        f'input cosmo ({cosmo_desc}) differs from gcdata cosmo ({cosmo_gcdata})')
            self.meta.__setitem__('cosmo', cosmo_desc, force=True)

    def update_cosmo(self, cosmo, overwrite=False):
        r"""Updates cosmo metadata if not present

        Parameters
        ----------
        cosmo: clmm.Cosmology
            Cosmology
        overwrite: bool
            Overwrites the current cosmo metadata. If false raises Error when cosmologies are
            different.

        Returns
        -------
        None
        """
        self.update_cosmo_ext_valid(self, cosmo, overwrite=overwrite)
