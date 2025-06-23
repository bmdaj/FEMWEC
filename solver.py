from discretization import dis
from physics import phy
from filter_threshold import filter_threshold
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot import *
import warnings


class eigen_top_opt_2D:
    """
    Main class for the 2D Topology Optimization framework in 2D.
    It may be used to:
    a) Run the forward problem for a given configuration of the dielectric function.
    b) Run an inverse design problem using the Topology Optimization framework. 
    """
    def __init__( self,
                  nElx, 
                  nEly,
                  dVini,
                  n_metal,
                  k_metal,
                  n_back,
                  k_back,
                  wl,
                  fR,
                  eta,
                  beta,
                  scaling,
                  alpha = 0.0):
        """
        Initialization of the main class.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ dVini: Initial value for the design variables.
        @ n_metal : Value for the refractive index of the metal.
        @ k_metal : Value for the exctinction coefficient of the metal.
        @ n_back: Value for the refractive index of the backgroud medium.
        @ k_clad: Value for the extinction coefficient of the background medium.
        @ wl: Wavelength of the problem (Frequency domain solver).
        @ fR: Filtering radius.
        @ eta: parameter that controls threshold value.
        @ beta: parameter that controls the threshold sharpness.
        @ scaling: Length scale of the problem.
        """
        warnings.filterwarnings("ignore") # we eliminate all possible warnings to make the notebooks more readable.

        self.nElX = nElx
        self.nElY = nEly
        
        self.mu = 1 # non-magnetic media
        self.wavelength = wl
        self.fR = fR
        self.dVini = dVini

        self.dis_0 = None
        self.dVs = None
        self.eta = eta
        self.beta = beta

        self.n_metal = n_metal
        self.n_back = n_back

        self.k_metal = k_metal
        self.k_back = k_back

        self.alpha = alpha # pamping factor

        # -----------------------------------------------------------------------------------
        # PHYSICS OF THE PROBLEM
        # ----------------------------------------------------------------------------------- 
        self.scaling = scaling # We give the scaling of the physical problem; i.e. 1e-9 for nm.

        self.phys = phy(self.n_metal,
                        self.k_metal,
                        self.n_back,
                        self.k_back,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        alpha = self.alpha
                        ) 
        

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE PROBLEM
        # -----------------------------------------------------------------------------------                 
        self.dis_0 = dis(self.scaling,
                    self.nElX,
                    self.nElY)

        # We set the indexes of the discretization: i.e. system matrix, boundary conditions ...

        self.dis_0.index_set() 
        # -----------------------------------------------------------------------------------  
        # FILTERING AND THRESHOLDING 
        # -----------------------------------------------------------------------------------   
        self.filThr =  filter_threshold(self.fR, self.nElX, self.nElY, self.eta, self.beta) 

        # -----------------------------------------------------------------------------------  
        # INITIALIZING DESIGN VARIABLES
        # -----------------------------------------------------------------------------------  
        self.dVs = self.dVini 

    

    def solve_forward(self, dVs):
        """
        Function to solve the forward FEM problem in the frequency domain given a distribution of dielectric function in the simulation domain.
        """

        E = self.dis_0.FEM_sol(dVs, self.phys, self.filThr)

        return E
    
    def return_mode(self, eigval_num):

        E = self.dis_0.get_fields(self.phys, eigval_num)
        Ex = E [:self.dis_0.nodesX*self.dis_0.nodesY]  
        Ey = E [self.dis_0.nodesX*self.dis_0.nodesY:2*self.dis_0.nodesX*self.dis_0.nodesY]  
        Ez = E [2*self.dis_0.nodesX*self.dis_0.nodesY:]

        return Ex, Ey, Ez  

    def plot_norm(self):
        """
        Plots the normalized electric field norm distribution.
        """
        
        plot_Enorm(self.dis_0)

    def plot_Efields(self):
        """
        Function to plot the electric and magnetic field components.
        """
        plot_E(self.dis_0)

    
    def plot_material(self):
        """
        Function to plot the electric and magnetic field components.
        """
        plot_mat(self.dis_0)