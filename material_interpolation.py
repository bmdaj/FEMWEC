import numpy as np 

def material_interpolation_linear(eps_r, x, alpha_i=0.0):
    """
    Function that implements the material interpolation for a semiconductor.
    It returns the interpolated field and its derivative with respect to the position.
    The non-physical imaginary part discourages intermediate values of the design variable.
    @ eps_r: relative permittivity of the semiconductor.
    @ x: position of the design variable.
    @ alpha_i: problem dependent sacling factor for the imaginary term.
    """
    eps_back = 1.0
    A = eps_back + x*(eps_r-eps_back) - 1j * alpha_i * x * (eps_back - x) #changed to account for silica cladding
    dAdx = (eps_r-eps_back) - 1j * alpha_i * (eps_back - 2*x)

    return A, dAdx

def material_interpolation_metal(n_metal, k_metal, n_back, k_back, x, alpha=0.0):
    """
    Function that implements the material interpolation for a metal.
    It returns the interpolated field and its derivative with respect to the position.
    It avoids artificial resonances in connection with transition from positive to negative dielectric index.
    @ n_metal: refractive index of the metal.
    @ k_metal: extinction cross section of the metal.
    @ n_back: Value for the refractive index of the background material.
    @ n_back: Value for the refractive index of the background material.
    @ x: value of the design variable
    """

    n_eff = n_back + x*(n_metal-n_back) # effective refractive index
    k_eff = 0 + x*(k_metal-k_back) # effective wavevector

    
    A = (n_eff**2-k_eff**2)-1j*(2*n_eff*k_eff) - 1j* alpha * x*(1-x)
    dAdx = 2*n_eff*(n_metal-n_back)-2*k_eff*(k_metal-k_back)-1j*(2*(n_metal-n_back)*k_eff+2*n_eff*(k_metal-k_back)) - 1j* alpha * (1 - 2* x)

    return A, dAdx

