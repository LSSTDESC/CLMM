"""
Define the custom data type
"""
import warnings
from collections import OrderedDict
from astropy.table import Table as APtable
import numpy as np

import qp


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
        if "cosmo" not in self:
            OrderedDict.__setitem__(self, "cosmo", None)

    def __setitem__(self, item, value):
        if item == "cosmo" and self.get("cosmo", None):
            raise ValueError(
                "cosmo must be changed via update_cosmo or update_cosmo_ext_valid method"
            )
        OrderedDict.__setitem__(self, item, value)

    def __getitem__(self, item):
        """
        Make class accept all letter casings
        """
        if isinstance(item, str):
            item = {n.lower(): n for n in self.keys()}[item.lower()]
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
        metakwargs = kwargs["meta"] if "meta" in kwargs else {}
        metakwargs = {} if metakwargs is None else metakwargs
        self.meta = GCMetaData(**metakwargs)
        # this attribute is set when source galaxies have p(z)
        self.pzpdf_info = {
            "type": None,
            "unpack_quantile_zbins_limits": (0, 5, 501),
        }

    def _str_colnames(self):
        """Colnames in comma separated str"""
        return ", ".join(self.colnames)

    def _str_meta_(self):
        """Metadata in comma separated str"""
        return ", ".join([f"{key}={value!r}" for key, value in self.meta.items()])

    def _str_pzpdf_info(self):
        out = self.pzpdf_info["type"]
        if out is not None:
            if out == "shared_bins":
                default_cfg = np.get_printoptions()  # keep default values
                np.set_printoptions(edgeitems=5, threshold=10)
                out += " " + str(np.round(self.pzpdf_info.get("zbins"), 2))
                np.set_printoptions(**default_cfg)
            elif out == "quantiles":
                np.set_printoptions(formatter={'float': "{0:g}".format}, edgeitems=3, threshold=6)
                out += " " + str(self.pzpdf_info["quantiles"])
                out += " - unpacked with zgrid : " + str(
                    self.pzpdf_info["unpack_quantile_zbins_limits"]
                )
        return out

    def __repr__(self):
        """Generates string for repr(GCData)"""
        description = [self._str_meta_(), "columns: " + self._str_colnames()]
        if self.pzpdf_info["type"]:
            description.append(f"pzpdf: {self.pzpdf_info['type']}")
        return f'{self.__class__.__name__}({", ".join(description)})'

    def __str__(self):
        """Generates string for print(GCData)"""
        out = [
            f"{self.__class__.__name__}",
            f"> defined by: {self._str_meta_()}",
            f"> with columns: {self._str_colnames()}",
            f"> {len(self)} objects",
            f"{APtable.__str__(self)}",
        ]
        if self.pzpdf_info["type"]:
            out.insert(3, f"> and pzpdf: {self._str_pzpdf_info()}")
        return "\n".join(out)

    def _html_table(self):
        """Get html table for display"""
        return "</i>".join(APtable._repr_html_(self).split("</i>")[1:])

    def _repr_html_(self):
        """Generates string for display(GCData)"""
        out = [
            f"<b>{self.__class__.__name__}</b>",
            f"<br> <b>defined by:</b> {self._str_meta_()}",
            f"<br> <b>with columns:</b> {self._str_colnames()}",
            f"<br> {len(self)} objects",
            f"<br> {self._html_table()}",
        ]
        if self.pzpdf_info["type"]:
            out.insert(3, f"<br> <b>and pzpdf:</b> {self._str_pzpdf_info()}")
        return "\n".join(out)

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
            name_dict = {n.lower(): n for n in self.colnames}
            item = item.lower()
            item = ",".join([name_dict[i] for i in item.split(",")])
        out = APtable.__getitem__(self, item)
        # sub cols or sub rows
        if not isinstance(item, (str, int, np.int64)):
            out.pzpdf_info = self.pzpdf_info
        return out

    def update_info_ext_valid(self, key, gcdata, ext_value, overwrite=False):
        r"""Updates cosmo metadata if the same as in gcdata

        Parameters
        ----------
        key: str
            Name of key to compare and update.
        gcdata: GCData
            Table to check if same cosmology and ensemble bins.
        ext_value:
            Value to be compared to.
        overwrite: bool
            Overwrites the current metadata. If false raises Error when values are different.

        Returns
        -------
        None
        """
        if ext_value:
            in_value = gcdata.meta[key]
            if in_value and in_value != ext_value:
                if overwrite:
                    warnings.warn(
                        f"input '{key}' ({ext_value}) overwriting gcdata '{key}' ({in_value})"
                    )
                else:
                    raise ValueError(
                        f"input '{key}' ({ext_value}) differs from gcdata '{key}' ({in_value})"
                    )
            OrderedDict.__setitem__(self.meta, key, ext_value)

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
        self.update_info_ext_valid("cosmo", gcdata, cosmo_desc, overwrite=overwrite)

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

    def has_pzpdfs(self):
        """Get pzbins and pzpdfs of galaxies

        Returns
        -------
        pzbins : array
            zbins of each object in data
        pzpdfs : array
            PDF of each object in data
        """
        pzpdf_type = self.pzpdf_info["type"]
        if pzpdf_type is None:
            return False
        if pzpdf_type == "shared_bins":
            return ("zbins" in self.pzpdf_info) and ("pzpdf" in self.columns)
        if pzpdf_type == "individual_bins":
            return ("pzbins" in self.columns) and ("pzpdf" in self.columns)
        if pzpdf_type == "quantiles":
            return ("quantiles" in self.pzpdf_info) and ("pzquantiles" in self.columns)
        raise NotImplementedError(f"PDF use '{pzpdf_type}' not implemented.")

    def get_pzpdfs(self):
        """Get pzbins and pzpdfs of galaxies

        Returns
        -------
        pzbins : array
            zbins of PDF. 1D if `shared_bins` or `quantiles`.
            zbins of each object in data if `individual_bins`.
        pzpdfs : array
            PDF of each object in data

        Notes
        -----
        If pzpdf type is quantiles, a pdf will be unpacked on a grid contructed with
        `np.linspace(*self.pzpdf_info["unpack_quantile_zbins_limits"])`
        """
        pzpdf_type = self.pzpdf_info["type"]
        if pzpdf_type is None:
            raise ValueError("No PDF information stored!")
        if pzpdf_type == "shared_bins":
            pzbins = self.pzpdf_info["zbins"]
            pzpdf = self["pzpdf"]
        elif pzpdf_type == "individual_bins":
            pzbins = self["pzbins"]
            pzpdf = self["pzpdf"]
        elif pzpdf_type == "quantiles":
            pzbins = np.linspace(*self.pzpdf_info["unpack_quantile_zbins_limits"])
            qp_ensemble = qp.Ensemble(
                qp.quant,
                data={
                    "quants": np.array(self.pzpdf_info["quantiles"]),
                    "locs": self["pzquantiles"],
                },
            )
            pzpdf = qp_ensemble.pdf(pzbins)
        else:
            raise NotImplementedError(f"PDF use '{pzpdf_type}' not implemented.")
        return pzbins, pzpdf
