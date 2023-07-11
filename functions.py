from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np

def conv2(x, y, mode='same'):
    """
    Python analogue to the Matlab conv2(A,B) function. Returns the two-dimensional convolution of matrices A and B.
    @ x: input matrix 1
    @ y: input matrix 2
    """
    #return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
    return convolve2d(x,  y, mode=mode)

def p_norm(vec, p):

    norm = np.power(np.sum(np.power(vec, p)),  1/p) 

    return norm

def sens_p_norm(vec, vec_sens, p):

    factor = np.array([vec[i]**(p-1.) * vec_sens [i, :]  for i in range(len(vec))])

    sens =  (np.sum(vec**p))**((1./p)-1.) * np.sum(factor, axis=0)

    return sens

def FOM_sens_log (FOM, sens):

    FOM_new = np.log10(FOM)
    sens_new = (1/(FOM*np.log(10))) * sens
    return FOM_new, sens_new


def FOM_sens_division(FOM, sens):

    FOM_new =  1 / FOM
    sens_new = - (1/FOM**2) * sens

    return FOM_new, sens_new

def FOM_sens_scaling(FOM, sens, scaling):

    FOM_new =   FOM * scaling
    sens_new = scaling * sens

    return FOM_new, sens_new


def effective_index(E, H, n, phy):
    # SOLVE INTEGRAL WITH SHAPE FUNCTIONS!!

    #fig, ax = plt.subplots(figsize=(14,10))
    #extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    #nominator =np.real((np.conj(H[:,1])*E[:,0]-np.conj(H[:,0])*E[:,1])) #* phy.scale**2
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #im = ax.imshow(np.reshape(nominator, (121,201)),cmap='inferno')#, origin="lower", extent=extent)#, vmax=4e5, cmap='inferno')
    #eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    #ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    #fig.colorbar(im, cax=cax, orientation='vertical')
    #ax.set_xlabel('$X (1 \\mu m)$')
    #ax.set_ylabel('$Y (1 \\mu m)$')
    #raise()
    nominator = np.sum(n**2*np.real(np.conj(H)[:,1]*E[:,0]-np.conj(H)[:,0]*E[:,1])) #* phy.scale**2
    #print(nominator)
    #plt.imshow(np.reshape(np.conj(H)[:,1]*E[:,0]-np.conj(H)[:,0]*E[:,1], (121,201)))
    #plt.show()
    #raise()
    #print(np.max((np.cross(abs(E),np.conj(abs(H)))[:,2]))* phy.scale**2)
    #print( nominator)
    #normE_2 = (E[:,0]*np.conj(E[:,0]) + E[:,1]*np.conj(E[:,1]) + E[:,2]*np.conj(E[:,2]))  #* phy.scale**2
    #normEz = np.sqrt(E[:,2]*np.conj(E[:,2]))
    #nominator = np.sum(n**2*normE_2)
    #denominator = np.sum(normE_2)
    #denominator = np.sum(n**2*normE_2)
    #print(denominator)
    #raise()
    normE_2 = (E[:,0]*np.conj(E[:,0]) + E[:,1]*np.conj(E[:,1]) + E[:,2]*np.conj(E[:,2])) #* phy.scale**2
    #normEz = np.sqrt(E[:,2]*np.conj(E[:,2]))
    #nominator = np.sum(n**2*normE_2)
    #denominator = np.sum(normE_2)
    denominator = np.sum(n**2*normE_2) # phy.scaling**2
    #print(denominator)
    
    neff = ( phy.mu_0 * phy.c ) * nominator / denominator
    #neff = nominator / denominator
    print("Calculated effective index: ", neff)
    
    return neff