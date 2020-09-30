# CLMModeling abstract class

class CLMModeling:
    def set_cosmo_params_dict (self, cosmo_dict):
        raise NotImplementedError

    def set_halo_density_profile (self, halo_profile_model = 'nfw', massdef = 'mean', delta_mdef = 200):
        raise NotImplementedError

    def set_concentration (self, cdelta):
        raise NotImplementedError

    def set_mass (self, mdelta):
        r""" Sets the value of the :math:`M_\Delta` 
        
        Parameters
        ----------
        mdelta : float
            Galaxy cluster mass :math:`M_\Delta` in units of :math:`M_\odot h^{-1}`
        
        """
        raise NotImplementedError

    def eval_da_z1z2 (self, z1, z2):
        raise NotImplementedError

    def eval_sigma_crit (self, z_len, z_src):
        raise NotImplementedError

    def eval_density (self, r3d, z_cl):
        raise NotImplementedError

    def eval_sigma (self, r_proj, z_cl):
        raise NotImplementedError

    def eval_sigma_mean (self, r_proj, z_cl):
        raise NotImplementedError

    def eval_sigma_excess (self, r_proj, z_cl):
        raise NotImplementedError

    def eval_shear (self, r_proj, z_cl, z_src):
        raise NotImplementedError
    