from clmm import Cosmology

# dc2_cosmo = Cosmology(H0=70.0, Omega_dm0=0.27 - 0.045, Omega_b0=0.045, Omega_k0=0.0)
dc2_cosmo = Cosmology(H0=71.0, Omega_dm0=0.265 - 0.0448, Omega_b0=0.0448, Omega_k0=0.0)

paper_modeling = {
    "massdef": "mean",
    "delta_mdef": 200,
    "halo_profile_model": "nfw",
}
paper_halo = {"mdelta": 1e14, "cdelta": 4, "z_cl": 0.4, **paper_modeling}
