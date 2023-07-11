import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def resize(A, nElx, nEly):
    import cv2
    A_T = np.reshape(np.real(A), (nEly, nElx)).T
    B_T = cv2.resize(A_T, dsize=(nEly+1, nElx+1), interpolation=cv2.INTER_AREA)
    return B_T.T

def init_plot_params(fontsize):
    """
    Initialization of the plottings style used in different plotting routines.
    @ fontsize: Font size
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("science")
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": fontsize})

def plot_Enorm(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.normE), (dis.nodesY, dis.nodesX)),cmap='inferno', origin="lower", extent=extent)#, origin="lower", extent=extent)#, vmax=4e5, cmap='inferno')
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Ez(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.Ez), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Ex(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    fig, ax = plt.subplots(figsize=(14,10))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.Ex), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)#, vmin=-1e5, vmax=1e5, cmap='inferno')
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Ex_im(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    fig, ax = plt.subplots(figsize=(14,10))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.imag(dis.Ex), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)#, vmin=-1e5, vmax=1e5, cmap='inferno')
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Ey(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.Ey), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)#vmin=-1e5, vmax=1e5, cmap='inferno')
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Hz(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.Hz), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Hx(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.Hx), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_Hy(dis):
    """
    Plots the electric field intensity for the whole simulation domain.
    """
    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.imag(dis.Hy), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_T(dis):

    init_plot_params(28)

    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.reshape(np.real(dis.T), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower", extent=extent)
    eps = resize(np.reshape(np.real(dis.A), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')

    plt.show()

def plot_mi(dis,  idx_wg, save, dir):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(40)
    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    wg_region = np.zeros((dis.nEly, dis.nElx))
    wg_region[idx_wg[0,:], idx_wg[1,:]] = 1

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.real(dis.dFPST), cmap='binary', origin="lower", vmax=1, vmin=0, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
    #print('Maximum real dielectric permittivity: ', np.max(np.real(dis.eps)))
    #print('Minimum real dielectric permittivity: ', np.min(np.real(dis.eps)))
    ax.set_xlabel('$X (\\mu$m)')
    ax.set_ylabel('$Y (\\mu$m)')
    plt.show()

    if save:
        fig.savefig(dir+"/material_distribution.svg")

def plot_mi_imag(dis):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(resize(np.imag(dis.dFPST), dis.nElx, dis.nEly), cmap='binary', origin="lower", vmax=1, vmin=0)
    eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    ax.contour(np.real(eps), levels=1, cmap='binary', linewidth=2, alpha=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    print('Maximum imaginary dielectric permittivity: ', np.max(np.imag(dis.eps)))
    print('Minimum imaginary dielectric permittivity: ', np.min(np.imag(dis.eps)))
    ax.set_xlabel('$x (nm)$')
    ax.set_ylabel('$y (nm)$')

    plt.show()

def plot_perm(dis,  idx_wg):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    wg_region = np.zeros((dis.nEly, dis.nElx))
    wg_region[idx_wg[0,:], idx_wg[1,:]] = 1

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(np.real(dis.eps), origin="lower", extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
    #print('Maximum real dielectric permittivity: ', np.max(np.real(dis.eps)))
    #print('Minimum real dielectric permittivity: ', np.min(np.real(dis.eps)))
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')
    plt.show()

def plot_perm_wg(dis,  idx_wg):
    """
    Plots the material interpolation for the whole simulation domain.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))
    extent = [-0.5*dis.nElx*dis.scaling * 1e6, 0.5*dis.nElx*dis.scaling * 1e6, -0.5*dis.nEly*dis.scaling * 1e6, 0.5*dis.nEly*dis.scaling * 1e6]
    wg_region = np.zeros((dis.nEly, dis.nElx))
    wg_region[idx_wg[0,:], idx_wg[1,:]] = 1

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    eps_wg = np.zeros_like(dis.eps)
    eps_wg[idx_wg[0,:], idx_wg[1,:]] = dis.eps[idx_wg[0,:], idx_wg[1,:]]
    im = ax.imshow(np.real(eps_wg), origin="lower", extent=extent, vmin=np.min(np.real(eps_wg[idx_wg[0,:], idx_wg[1,:]])))
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
    #print('Maximum real dielectric permittivity: ', np.max(np.real(dis.eps)))
    #print('Minimum real dielectric permittivity: ', np.min(np.real(dis.eps)))
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')
    plt.show()

def plot_iteration(dis):
    """
    Plots the material interpolation and the electric field intensity for the whole simulation domain.
    Applied in each iteration of the optimization.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(1,2,figsize=(20,4))
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    design_field =  np.reshape(np.real(dis.dFPST), (dis.nodesY-1, dis.nodesX-1))
    im0 = ax[0].imshow(design_field, vmin=0, vmax=1, aspect='auto', origin="lower", cmap='binary')
    #im =  ax[1].imshow(np.reshape(np.real(dis.normE), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower")
    im =  ax[1].imshow(np.reshape(np.real(dis.normE), (dis.nodesY, dis.nodesX)), cmap='inferno', origin="lower")
    fig.colorbar(im0, cax=cax, orientation='vertical')

    for axis in ax:
        axis.set_xlabel('$X (0.1 \\mu m)$')
        axis.set_ylabel('$Y (0.1 \\mu m)$')
    plt.show()

def plot_sens(dis, idxdr, sens):
    """
    Plots the sensitivities for the design region.
    """
    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #sens = np.ones_like(sens)
    region = np.zeros((dis.nEly, dis.nElx))
    region [idxdr[0,:], idxdr[1,:]] = np.reshape(sens, np.shape(region [idxdr[0,:], idxdr[1,:]]))

    im = ax.imshow(np.real(region), cmap='inferno')#, vmax=0.0025, vmin=-0.001)
    #ax.contour(np.real(dis.eps), levels=2, cmap='binary', linewidth=2, alpha=1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('$x (nm)$')
    ax.set_ylabel('$y (nm)$')

    plt.show()

def plot_it_history(maxItr, FOM_list, heat_const_list, heat_cons_val, it_num):

    iterations = np.linspace(0,maxItr-1, maxItr) [:it_num]

    init_plot_params(24)
    fig, ax = plt.subplots(1,2,figsize=(16,6))

    ax[0].set_ylabel("FOM")
    ax[1].set_ylabel("Volume fraction")

    ax[0].scatter(iterations, FOM_list [:it_num], color='blue')
    ax[0].plot(iterations, FOM_list[:it_num], color='blue',  alpha=0.5)
    #ax[0].plot(iterations, np.ones_like(iterations), color='gray', alpha=1, linestyle="dashed", label="No-metal solution")

    ax[1].scatter(iterations, heat_const_list [:it_num], color='red')
    ax[1].plot(iterations, heat_const_list [:it_num], color='red',  alpha=0.5, label="Optimization")

    ax[1].plot(iterations, np.ones_like(iterations)*heat_cons_val, color='red', alpha=0.8, linestyle="dashed", label="Minimum volume fraction")
    ax[1].legend(frameon=True, fontsize=20)
    ax[0].legend(frameon=True, fontsize=20)
    ax[1].set_ylim(0,1)
    ax[0].set_yscale("log")
    
    for axis in ax:
        axis.set_xlabel("Iteration number")

    plt.show()

def plot_it_history_minmax(maxItr, FOM_list, constraint1_list, constraint2_list, constraint3_list, it_num, save, dir):

    iterations = np.linspace(0,maxItr-1, maxItr) [:it_num]

    init_plot_params(24)
    fig, ax = plt.subplots(1,3,figsize=(30,6))

    ax[0].set_ylabel("FOM")
    ax[1].set_ylabel("FOMs")
    ax[2].set_ylabel("Volume constraint")
    ax[0].scatter(iterations, FOM_list [:it_num], color='k')
    ax[0].plot(iterations, FOM_list[:it_num], color='k',  alpha=0.5)
    #ax[0].plot(iterations, np.ones_like(iterations), color='gray', alpha=1, linestyle="dashed", label="No-metal solution")
    ax[1].scatter(iterations, constraint1_list [:it_num], color='b')
    ax[1].plot(iterations, constraint1_list [:it_num], color='b',  alpha=0.5, label="Unheated device")
    ax[1].scatter(iterations, constraint2_list [:it_num], color='r')
    ax[1].plot(iterations, constraint2_list [:it_num], color='r',  alpha=0.5, label="Heated device")
    ax[2].scatter(iterations, constraint3_list [:it_num], color='g')
    ax[2].plot(iterations, constraint3_list [:it_num], color='g',  alpha=0.5)
    #if hasattr(constraint3_list, "__len__"):
    #    ax[1].scatter(iterations, constraint3_list [:it_num], color='g')
    #    ax[1].plot(iterations, constraint3_list [:it_num], color='g',  alpha=0.5)
    #ax[1].plot(iterations, np.ones_like(iterations)*heat_cons_val, color='red', alpha=0.8, linestyle="dashed", label="Minimum volume fraction")
    ax[2].legend(frameon=True, fontsize=20)
    ax[1].legend(frameon=True, fontsize=20)
    ax[0].legend(frameon=True, fontsize=20)
    #ax[1].set_ylim(0,1)
    #ax[0].set_yscale("log")
    #ax[1].set_yscale("log")
    
    for axis in ax:
        axis.set_xlabel("Iteration number")

    plt.show()

    if save:
        fig.savefig(dir+"/iteration_history.svg")

def plot_lam_minmax(maxItr, opt, it_num, save, dir):

    iterations = np.linspace(0,maxItr-1, maxItr) [1:it_num]

    init_plot_params(24)
    fig, ax = plt.subplots(1,2,figsize=(18,6))
    for axis in ax:
        axis.set_ylabel("$\\lambda$")
    ax[0].scatter(iterations, opt.lam_array [0, 1:it_num], color='b')
    ax[0].plot(iterations, opt.lam_array [0, 1:it_num], color='b',  alpha=0.5, label="Unheated device")
    ax[0].scatter(iterations, opt.lam_array [1, 1:it_num], color='r')
    ax[0].plot(iterations, opt.lam_array [1, 1:it_num], color='r',  alpha=0.5, label="Heated device")
    ax[1].scatter(iterations, opt.lam_array [2, 1:it_num], color='g')
    ax[1].plot(iterations, opt.lam_array [2, 1:it_num], color='g',  alpha=0.5, label="Volume constraint")
    ax[0].legend(frameon=True, fontsize=20)
    ax[1].legend(frameon=True, fontsize=20)

    for axis in ax: 
        axis.set_xlabel("Iteration number")

    plt.show()

    if save:
        fig.savefig(dir+"/lambda_history.svg")

def plot_geom_minmax(maxItr, opt, it_num, save, dir):

    iterations = np.linspace(0,maxItr-1, maxItr) [1:it_num]

    init_plot_params(24)
    fig, ax = plt.subplots(figsize=(9,6))
    ax.set_ylabel("Geometric constraints")
    ax.scatter(iterations, opt.cons_4_it [1:it_num], color='b')
    ax.plot(iterations, opt.cons_4_it [1:it_num], color='b',  alpha=0.5, label="Solid constraint")
    ax.scatter(iterations, opt.cons_5_it [1:it_num], color='r')
    ax.plot(iterations, opt.cons_5_it [1:it_num], color='r',  alpha=0.5, label="Void constraint")
    ax.legend(frameon=True, fontsize=20)

    ax.set_xlabel("Iteration number")

    plt.show()

    if save:
        fig.savefig(dir+"/geom_history.svg")


def plot_change_dVs(delta_dV, delta_dV_real, dis, idxdr):

    init_plot_params(28)
    fig, ax = plt.subplots(1,2, figsize=(22,10))
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    region_dV = np.zeros((dis.nEly, dis.nElx))
    region_dV [idxdr[0,:], idxdr[1,:]] = np.reshape(delta_dV, np.shape(region_dV [idxdr[0,:], idxdr[1,:]]))

    im = ax[0].imshow(np.real(region_dV), cmap='binary')
    im1 = ax[1].imshow(np.real(delta_dV_real), cmap='binary')
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.colorbar(im1, cax=cax1, orientation='vertical')
    

    for axis in ax:
        axis.set_xlabel('$x (nm)$')
        axis.set_ylabel('$y (nm)$')

    plt.show()

def  plot_frequency_response(wl_array, wl_high_array, wl_cen, vals, scaling):

    init_plot_params(24)
    fig, ax = plt.subplots(figsize=(14,10))

    ax.set_ylabel("FOM")
    ax.set_xlabel("$\\lambda$ ($\\mu$m)")

    #ax.scatter(wl_array*scaling*1E6, vals[:len(wl_array)], color="blue")
    ax.scatter(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple")

    #ax.plot(wl_array*scaling*1E6, vals[:len(wl_array)], color="blue",alpha=0.8, linestyle="dashed", label="Bandwidth response")
    ax.plot(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple",alpha=0.8, linestyle="dashed")#, label="Highly resolved central region")

    plt.axvline(x = wl_cen*scaling*1E6, color = 'black', alpha=0.25, linestyle="dashed", label = '$\\lambda_\\text{cen}$')
    
    ax.legend(frameon=True)

    plt.show()

def  plot_propagation_response(neff_array, neff0, neffT, vals, save=False, dir=None):

    init_plot_params(24)
    fig, ax = plt.subplots(figsize=(14,10))

    ax.set_ylabel("FOM")
    ax.set_xlabel("$n_\\text{eff}=k_z / k_0$")

    ax.scatter(neff_array, vals, color="black")
    #ax.scatter(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple")

    ax.plot(neff_array, vals, color="black",alpha=0.8, linestyle="dashed")
    #ax.plot(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple",alpha=0.8, linestyle="dashed")#, label="Highly resolved central region")

    #plt.axvline(x = neff0, color = 'blue', alpha=0.25, linestyle="dashed", label = '$n_\\text{eff} (T=T_0)$')
    #plt.axvline(x = neffT, color = 'red', alpha=0.25, linestyle="dashed", label = '$n_\\text{eff} (T=T_1)$')
    
    #ax.legend(frameon=True)

    ax.set_yscale("log")

    if save:
        fig.savefig(dir+"/heated_response.svg")

    plt.show()

def  plot_propagation_response_coupled(neff_array, neff0, neffT, vals, vals_heated, save=False, dir=None):

    init_plot_params(24)
    fig, ax = plt.subplots(figsize=(14,10))

    ax.set_ylabel("FOM")
    ax.set_xlabel("$n_\\text{eff}=k_z / k_0$")

    ax.scatter(neff_array, vals, color="blue")
    #ax.scatter(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple")

    ax.plot(neff_array, vals, color="blue",alpha=0.8, linestyle="dashed")
    #ax.plot(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple",alpha=0.8, linestyle="dashed")#, label="Highly resolved central region")

    ax.scatter(neff_array, vals_heated, color="red")
    #ax.scatter(wl_high_array*scaling*1E6, vals[len(wl_array):], color="purple")

    ax.plot(neff_array, vals_heated, color="red",alpha=0.8, linestyle="dashed")

    plt.axvline(x = neff0, color = 'blue', label = '$n_\\text{eff} (T=T_0)$')
    plt.axvline(x = neffT, color = 'red', label = '$n_\\text{eff} (T=T_1)$')
    
    ax.legend(frameon=True)

    ax.set_yscale("log")

    plt.show()

    if save:
        fig.savefig(dir+"/coupled_response.svg")

def plot_n(n,  dis, nElYcore, nElXcore):

    """
    Plots the material interpolation for the whole simulation domain.
    """

    index_wg = dis.indexes_FOM

    init_plot_params(28)
    fig, ax = plt.subplots(figsize=(14,10))
    extenty = nElYcore * dis.scaling
    extentx = nElXcore * dis.scaling
    print(extentx, extenty)
    extent = [-0.5*extentx *1e6, 0.5*extentx *1e6, -0.5*extenty *1e6, 0.5*extenty *1e6]
    print(np.shape(n))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    n_wg = np.zeros((nElYcore, nElXcore))
    n_wg = np.reshape(np.real(n)[index_wg[0,:], index_wg[1,:],0], np.shape(n_wg))
    im = ax.imshow(n_wg, cmap='viridis', origin="lower", extent=extent, aspect='equal')
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$X (\\mu m)$')
    ax.set_ylabel('$Y (\\mu m)$')
    plt.show()