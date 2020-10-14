# CLMModeling abstract class

class CLMModeling:
    def set_cosmo_params_dict(self, cosmo_dict):
        raise NotImplementedError

    def set_halo_density_profile(self, halo_profile_model='nfw', massdef='mean', delta_mdef=200):
        raise NotImplementedError

    def set_concentration(self, cdelta):
        raise NotImplementedError

    def set_mass(self, mdelta):
        r""" Sets the value of the :math:`M_\Delta` 
        
        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot h^{-1}`
        
        """
        raise NotImplementedError

    def eval_da_z1z2(self, z1, z2):
        r"""Calculate the angular diameter distance between two scale factors.
        
        Parameters
        ----------
        z1 : float
            Redshift.
        z2 : float, optional
            Redshift.
        
        Returns
        -------
        float
            Angular diameter distance in units :math:`M\!pc\ h^{-1}`
        """
        raise NotImplementedError

    def eval_sigma_crit(self, z_len, z_src):
        r"""Computes the critical surface density
        
        Parameters
        ----------
        z_len : float
            Lens redshift
        z_src : array_like, float
            Background source galaxy redshift(s)
        
        Returns
        -------
        float
            Cosmology-dependent critical surface density in units of :math:`h\ M_\odot\ pc^{-2}`
        """
        raise NotImplementedError

    def eval_density(self, r3d, z_cl):
        r"""Retrieve the 3d density :math:`\rho(r)`.
        
        Parameters
        ----------
        r3d : array_like, float
            Radial position from the cluster center in :math:`M\!pc\ h^{-1}`.
        z_cl: float
            Redshift of the cluster
        
        Returns
        -------
        array_like, float
            3-dimensional mass density in units of :math:`h^2\ M_\odot\ pc^{-3}` DOUBLE CHECK THIS
        """
        raise NotImplementedError

    def eval_sigma(self, r_proj, z_cl):
        r""" Computes the surface mass density
        
        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        
        Returns
        -------
        array_like, float
            2D projected surface density in units of :math:`M_\odot\ Mpc^{-2}`
        """
        raise NotImplementedError

    def eval_sigma_mean(self, r_proj, z_cl):
        raise NotImplementedError

    def eval_sigma_excess(self, r_proj, z_cl):
        r""" Computes the excess surface density
        Parameters
        ----------
        r_proj : array_like
            Projected radial position from the cluster center in :math:`M\!pc`.
        z_cl: float
            Redshift of the cluster
        
        Returns
        -------
        array_like, float
            Excess surface density in units of :math:`M_\odot\ Mpc^{-2}`.
        """
        raise NotImplementedError

    def eval_shear(self, r_proj, z_cl, z_src):
        r"""Computes the tangential shear
        
        Parameters
        ----------
        r_proj : array_like
            The projected radial positions in :math:`M\!pc\ h^{-1}`.
        z_cl : float
            Galaxy cluster redshift
        z_src : array_like, float
            Background source galaxy redshift(s)
        
        Returns
        -------
        array_like, float
            tangential shear
        """
        raise NotImplementedError
    
