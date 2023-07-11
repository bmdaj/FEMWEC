import numpy as np
import matplotlib.pyplot as plt
def I_s(c, d_dFPS_2, dFPST):
    "Structural solid indicator"
    return dFPST * np.exp(-c*d_dFPS_2)

def I_v(c, d_dFPS_2, dFPST):
    "Structural void indicator"
    return (1.0-dFPST) * np.exp(-c*d_dFPS_2)

def g_s(dFPS, n, eta_e, c, d_dFPS_2, dFPST):
    "Geometric constraint for solid in lengthscale constraint"
    deriv_dFPS = d_dFPS_2

    return (1/n)* np.sum(I_s(c, deriv_dFPS, dFPST) * np.minimum((dFPS-eta_e), np.zeros_like(dFPST, dtype="complex128"))**2)

def g_v(dFPS, n, eta_d, c, d_dFPS_2,dFPST):
    "Geometric constraint for solid in lengthscale constraint"
    deriv_dFPS = d_dFPS_2

    return (1/n)* np.sum(I_v(c, deriv_dFPS, dFPST) * np.minimum((eta_d-dFPS), np.zeros_like(dFPST, dtype="complex128"))**2)

def sens_g_s(dFPS, n, eta_e, c, grad_dFPS_2, dFPST, grad_dFPS, d_thresh, d_thresh_filter, d_filter, d_grad_filter):

    def sens_I_s(c, grad_dFPS_2, d_thresh_filter, dFPST, grad_dFPS, d_grad_filter):

        factor_I = np.exp(-c*grad_dFPS_2)
        term_I_1 = d_thresh #d_thresh_filter
        term_I_2 = - 2 * c * dFPST * grad_dFPS_2 #grad_dFPS * d_grad_filter

        return factor_I * (term_I_1+term_I_2)

    factor_g = (1/n) 
    term_g_1 =  (dFPS - eta_e)**2 * sens_I_s(c, grad_dFPS_2, d_thresh_filter, dFPST, grad_dFPS, d_grad_filter) # * d_filter
    term_g_2 =  (dFPS - eta_e) * 2 * I_s(c, grad_dFPS_2, dFPST) #* d_filter

    index_condition = np.array(np.where(dFPS>eta_e))
    sens = factor_g * (term_g_1+term_g_2)

    sens [index_condition[0,:], index_condition[1,:]] = 0.0

    return sens

def sens_g_v(dFPS, n, eta_d, c, grad_dFPS_2, dFPST, grad_dFPS, d_thresh, d_thresh_filter, d_filter, d_grad_filter):

    def sens_I_v(c, grad_dFPS_2, d_thresh_filter, dFPST, grad_dFPS, d_grad_filter):

        factor_I = np.exp(-c*grad_dFPS_2)
        term_I_1 = - d_thresh #d_thresh_filter
        term_I_2 = - 2 * c * (1.0-dFPST) * grad_dFPS_2 #* grad_dFPS * d_grad_filter

        return factor_I * (term_I_1+ term_I_2)

    factor_g = (1/n) 
    term_g_1 =  (eta_d-dFPS)**2 * sens_I_v(c, grad_dFPS_2, d_thresh_filter, dFPST, grad_dFPS, d_grad_filter) 
    term_g_2 =  (eta_d-dFPS) * 2 * I_v(c, grad_dFPS_2, dFPST) #* d_filter

    index_condition = np.array(np.where(dFPS<eta_d))
    sens = factor_g * (term_g_1-term_g_2)

    sens [index_condition[0,:], index_condition[1,:]] = 0.0

    return sens