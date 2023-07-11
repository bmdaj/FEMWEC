import numpy as np 
import scipy
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt

def material_interpolation_sc(eps_r, x, alpha_i=1.0):
    """
    Function that implements the material interpolation for a semiconductor.
    It returns the interpolated field and its derivative with respect to the position.
    The non-physical imaginary part discourages intermediate values of the design variable.
    @ eps_r: relative permittivity of the semiconductor.
    @ x: position of the design variable.
    @ alpha_i: problem dependent sacling factor for the imaginary term.
    """
    eps_siO = 4.2
    A = eps_siO + x*(eps_r-eps_siO) - 1j * alpha_i * x * (eps_siO - x) #changed to account for silica cladding
    dAdx = (eps_r-eps_siO) - 1j * alpha_i * (eps_siO - 2*x)

    return A, dAdx

def material_interpolation_metal(n_metal, k_r, n_wg, n_clad, x, idx, idx_wg, delta_n, alpha=0.0):
    """
    Function that implements the material interpolation for a metal.
    It returns the interpolated field and its derivative with respect to the position.
    It avoids artificial resonances in connection with transition from positive to negative dielectric index.
    @ n_metal: refractive index of the metal.
    @ k_r: extinction cross section of the metal.
    @ n_wg: Value for the refractive index of the waveguide.
    @ n_clad: Value for the refractive index of the cladding.
    @ x: value of the design variable
    """

    #alpha = 0.0

    #index_change = np.zeros((1, len(delta_n.flatten())))
    #index_change[0,:] = delta_n.flatten()

    n_bck = n_clad * np.ones_like(x, dtype="complex128")
    #n_bck_T = n_clad * np.ones_like(delta_n, dtype="complex128")
    n_bck [idx_wg[0,:], idx_wg[1,:]] = n_wg
    #n_bck_T [idx_wg[0,:], idx_wg[1,:], :] = n_wg
    #print(np.shape(idx_wg[0,:]))
    #print(np.shape(idx_wg[1,:]))
    #print(np.shape(delta_n))
    #print(np.shape(np.ravel(delta_n)))
    #print(np.shape(idx_wg[0,:]))
    #n_bck [idx_wg[0,:], idx_wg[1,:]] += delta_n [idx_wg[0,:], idx_wg[1,:]] # change in refractive index due to temperature changeT
    if np.max(delta_n) != 0:
        n_bck [idx_wg[0,:], idx_wg[1,:]] += 0.25*np.sum(delta_n, axis=2)[idx_wg[0,:], idx_wg[1,:]] # change in refractive index due to temperature changeT
    #n_bck += 0.25*np.array(np.sum(delta_n, axis=2)) # change in refractive index due to temperature changeT
    #n_bck_T [idx_wg[0,:], idx_wg[1,:], :] += np.ravel(delta_n)

    n_eff = n_bck + x*(n_metal-n_bck) # effective refractive index
    k_eff = 0 + x*(k_r-0) # effective wavevector

    
    A = (n_eff**2-k_eff**2)-1j*(2*n_eff*k_eff) - 1j* alpha * x*(1-x)
    dAdx = 2*n_eff*(n_metal-n_bck)-2*k_eff*(k_r-0)-1j*(2*(n_metal-n_bck)*k_eff+2*n_eff*(k_r-0)) - 1j* alpha * (1 - 2* x)

    dndT_Si = 1.8E-4 
    dAdT = np.zeros_like(delta_n, dtype="complex128")
    #dAdT = np.zeros_like(x, dtype="complex128")

    if np.max(delta_n) != 0:
        for i in range(4):
            n_bck_T = n_clad * np.ones_like(x, dtype="complex128")
            n_bck_T [idx_wg[0,:], idx_wg[1,:]] = n_wg
            n_bck_T[idx_wg[0,:], idx_wg[1,:]] += delta_n[idx_wg[0,:], idx_wg[1,:], i] 
            n_eff_T = n_bck_T + x*(n_metal-n_bck_T)
            neffdT = dndT_Si * (1-x)
            #dAdT = (2 * neffdT * (n_eff-1j*k_eff))
            dAdT[idx_wg[0,:], idx_wg[1,:], i] = (2 * neffdT * (n_eff_T-1j*k_eff))[idx_wg[0,:], idx_wg[1,:]]

    #dndT_Si = 1.8E-4
    #neffdT = dndT_Si * (1-x)
    #dAdT = np.zeros_like(x, dtype="complex128")
    #dAdT = (2 * neffdT * (n_eff-1j*k_eff))
    #dAdT[idx_wg[0,:], idx_wg[1,:]] = (2 * neffdT * (n_eff-1j*k_eff))[idx_wg[0,:], idx_wg[1,:]]
    #plt.imshow(np.real(dAdT), vmin=np.min(np.real((2 * neffdT * (n_eff-1j*k_eff))[idx_wg[0,:], idx_wg[1,:]])))
    #print(np.max(dAdT))
    #print(np.min(dAdT))
    #print(np.min(np.real((2 * neffdT * (n_eff-1j*k_eff))[idx_wg[0,:], idx_wg[1,:]])))
    #plt.show()
    #raise()
    #dAdT[idx_wg[0,:], idx_wg[1,:]] = 2 * neffdT[idx_wg[0,:], idx_wg[1,:]] * (n_eff[idx_wg[0,:], idx_wg[1,:]]-1j*k_eff[idx_wg[0,:], idx_wg[1,:]])


    #if delta_n.all() != 0:
        #dAdx_heat = np.zeros_like(x)
        #deps_prime_prime = 2*((n_metal-n_bck)*(n_eff-1j*k_eff)-(k_r-0.0)*(k_eff+1j*n_eff)) 
        #dn_dT_Si = 1.8E-4
        #dAdx_heat = np.zeros_like(x.flatten())
        #AdjRHS_el = 2 * dn_dT_Si * (1-x) * (n_eff-1j*k_eff) # This is for elements and should be for nodes. Use edofmat to convert into nodes!
        #AdjRHS = np.zeros_like(dis_heat.T)
        #dSdx = np.zeros((len(dis_heat.T),len(dis_heat.T)))
        #for i in range(len(dis_heat.edofMat)):
            #nodes = (dis_heat.edofMat[i]).astype(int)
            #AdjRHS [nodes] += AdjRHS_el.flatten()[i]
            #dSdx [np.ix_(nodes, nodes)] = dis_heat.dAdx.flatten()[i] * dis_heat.LEM
        #AdjLambda_eps = dis_heat.PR.T @  scipy.sparse.linalg.spsolve(dis_heat.L.T, scipy.sparse.linalg.spsolve(dis_heat.U.T, dis_heat.PC.T @ (-AdjRHS)))
        #dAdx_heat = np.zeros_like(x.flatten())
        #plt.imshow(np.real(np.reshape(dis_heat.T, (dis_heat.nEly+1, dis_heat.nElx+1))))
        #plt.show()
        #raise()

        #for i in range(len(dis_heat.edofMat)):
            
            #dSdx_e = dis_heat.dAdx.flatten()[i] * dis_heat.LEM
            #AdjLambda_e = np.array([AdjLambda_eps[n] for n in dis_heat.edofMat[i].astype(int)])
            #T_e = np.array([dis_heat.T[n] for n in dis_heat.edofMat[i].astype(int)]).flatten()
            #dFdx_e =  np.array([dis_heat.dFdx_e[i], dis_heat.dFdx_e[i], dis_heat.dFdx_e[i], dis_heat.dFdx_e[i]])
            #dFdx_e =  np.array([dis_heat.dFdx[n] for n in dis_heat.edofMat[i].astype(int)]).flatten()
            #print(np.shape(deps_prime_prime.flatten()))
            #print(np.shape(AdjLambda_e[np.newaxis]))
            #print(np.shape(dSdx_e))
            #print(np.shape(T_e))
            #print(np.shape(dFdx_e))
            #print(np.shape(AdjLambda_eps[np.newaxis]))
            #print(np.shape(dis_heat.dFdx_e))deps_prime_prime.flatten() [i]

            #dAdx_heat [i] = deps_prime_prime.flatten() [i] + (np.real((AdjLambda_eps[np.newaxis] @ ((dSdx @ dis_heat.T) - dis_heat.dFdx)) [0]))  #-  np.real((AdjLambda_eps[np.newaxis] @ dis_heat.dFdx) [0]))
            #dAdx_heat [i] =   deps_prime_prime.flatten() [i] + (AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e - dFdx_e)))[0] -  (AdjLambda_eps[np.newaxis] @ dis_heat.dFdx) [0]

        #dAdx = np.reshape(dAdx_heat, np.shape(dAdx))

    return A, dAdx, dAdT

def material_interpolation_heat(k_metal, k_wg, k_clad, x, idx, alpha_i=0.0):

    k_bck = k_clad * np.ones_like(x, dtype="complex128")
    k_bck [idx[0,:], idx[1,:]] = k_wg

    p1 = 1

    A = k_bck + x**p1*(k_metal-k_bck) #- 1j * alpha_i * x * (1 - x) # changed to account for silica cladding
    dAdx = x**(p1-1) * (k_metal-k_bck) #- 1j * alpha_i * (1 - 2*x)

    return A, dAdx
