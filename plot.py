import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import resize

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

    fig, ax = plt.subplots(figsize=(3,3))
    extent = [-0.5*dis.nElx*dis.scaling, 0.5*dis.nElx*dis.scaling, -0.5*dis.nEly*dis.scaling, 0.5*dis.nEly*dis.scaling]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    normE = np.sqrt(np.real(dis.Ex)**2 + np.real(dis.Ey)**2 + np.real(dis.Ez)**2)

    im = ax.imshow(np.reshape(normE, (dis.nodesY, dis.nodesX)),cmap='inferno', origin="lower", extent=extent)#, origin="lower", extent=extent)#, vmax=4e5, cmap='inferno')
    #eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)
    #ax.contour(np.real(eps), levels=2, cmap='binary', linewidth=2, alpha=1, extent=extent)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')

    plt.show()

def plot_E(dis):
        """
        Plots the electric field components for the whole simulation domain.
        """
        init_plot_params(28)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        labels = ["$E_x$", "$E_y$", "$E_z$"]
        fields = [dis.Ex, dis.Ey, dis.Ez]
        extent = [-0.5 * dis.nElx * dis.scaling, 0.5 * dis.nElx * dis.scaling ,
                  -0.5 * dis.nEly * dis.scaling, 0.5 * dis.nEly * dis.scaling]
        #eps = resize(np.reshape(np.real(dis.eps), (dis.nEly, dis.nElx)), dis.nElx, dis.nEly)

        for i, ax in enumerate(axes):
            im = ax.imshow(np.reshape(np.real(fields[i]), (dis.nodesY, dis.nodesX)), cmap='seismic', origin="lower", extent=extent)
            #ax.contour(np.real(eps), levels=2, cmap='binary', linewidths=2, alpha=1, extent=extent)
            ax.set_title(labels[i])
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()
        plt.show()

def plot_mat(dis):
        """
        Plots the electric field components for the whole simulation domain.
        """
        init_plot_params(28)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        eps = dis.eps.reshape((dis.nEly, dis.nElx))
        dens = dis.dFPST.reshape((dis.nEly, dis.nElx))
        labels = ["$\epsilon$", "Density"]
        fields = [eps, dens]
        extent = [-0.5 * dis.nElx * dis.scaling, 0.5 * dis.nElx * dis.scaling,
              -0.5 * dis.nEly * dis.scaling, 0.5 * dis.nEly * dis.scaling]

        for i, ax in enumerate(axes):
            im = ax.imshow(np.real(fields[i]), cmap='viridis', origin="lower", extent=extent)
            ax.set_title(labels[i])
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$y$ (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()
        plt.show()



        

