import numpy as np

class phy:
    " Class that defines the physics of the model."

    def __init__( self,
                 n_metal,
                 k_r,
                 n_wg,
                 n_clad,
                 mu,
                 scale,
                 wavelength,
                 delta,
                 dz=None,
                 k_wg =None,
                 k_clad = None,
                 k_metal = None,
                 alpha = 1.0):
        """
        @ n_metal : Value for the refractive index of the metal.
        @ k_r : Value for the exctinction coefficient of the metal.
        @ n_wg: Value for the refractive index of the waveguide.
        @ n_clad: Value for the refractive index of the cladding.
        @ scale: scale of the physical problem; i.e. 1e-9 for nm.
        @ wavelength: Wavelength of the problem (Frequency domain solver).
        @ delta: Value of the effective refractive index (COMSOL).
        """

        # ------------------------------------------------------------------
        # ELECTROMAGNETICS 
        # ------------------------------------------------------------------

        self.n_metal = n_metal
        self.k_r = k_r
        self.eps_metal = n_metal**2

        self.n_wg = n_wg
        self.eps_wg =  n_wg**2

        self.n_clad = n_clad
        self.eps_clad =  n_clad**2

        self.mu = mu
        self.scale = scale
        self.wavelength = wavelength
        self.k = 2 * np.pi / (self.wavelength * self.scale)
        self.delta = delta
        self.kz = delta * self.k

        self.eps_0 = 8.85e-12
        self.mu_0 = 1.257e-6
        self.c = 3e8
        self.Z0 = np.sqrt(self.mu_0/self.eps_0)
        self.alpha = alpha


        # ------------------------------------------------------------------
        # HEAT
        # ------------------------------------------------------------------

        self.dz = dz
        self.k_metal = k_metal 
        self.k_clad = k_clad
        self.k_wg = k_wg

