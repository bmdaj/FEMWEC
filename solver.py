from discretization import dis
from heat_discretization import dis_heat
import numpy as np 
from physics import phy
from filter_threshold import filter_threshold
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plot import plot_Enorm, plot_Ez, plot_Ex, plot_Ey, plot_Hz, plot_mi, plot_mi_imag, plot_sens, plot_it_history, plot_frequency_response, plot_propagation_response
from plot import plot_Hx, plot_Hy, plot_n, plot_T, plot_it_history_minmax, plot_propagation_response_coupled, plot_perm, plot_perm_wg, plot_iteration, plot_Ex_im, plot_lam_minmax
from plot import plot_change_dVs, plot_geom_minmax
from functions import p_norm, sens_p_norm, FOM_sens_log, FOM_sens_division, FOM_sens_scaling
import warnings
import nlopt
import time 
from tqdm import tqdm
from logfile import create_logfile_optimization, create_logfile_sweep, init_dir
from optimization import optimizer
from constraints import g_s, g_v, sens_g_s, sens_g_v


def node_to_el(array, edofmat, nElx, nEly):
    array_el = np.zeros(nEly*nElx)
    vals_nodes = array[edofmat]
    vals_el = 0.25*np.sum(vals_nodes, axis=1)
    return vals_el

def resize_to_node(A, nElx, nEly):
    import cv2
    A_T = np.reshape(A, (nEly, nElx)).T
    B_T = cv2.resize(A_T, dsize=(nEly+1, nElx+1), interpolation=cv2.INTER_LINEAR)
    return B_T.T

class freq_top_opt_2D:
    """
    Main class for the 2D Topology Optimization framework in 2D.
    It may be used to:
    a) Run the forward problem for a given configuration of the dielectric function.
    b) Run an inverse design problem using the Topology Optimization framework. 
    """
    def __init__( self,
                  nElx_EM, 
                  nEly_EM,
                  nElx_heat,
                  nEly_heat,
                  dVini,
                  w_core,
                  h_core,
                  n_metal,
                  k_r,
                  n_wg,
                  n_clad,
                  wl,
                  delta,
                  deltaT,  
                  fR,
                  eta,
                  beta,
                  scaling,
                  vol_cons_val,
                  FOM_type, 
                  indexes_wg_EM,
                  indexes_wg_heat,
                  indexes_design_region_EM,
                  indexes_design_region_heat,
                  heat_cons_val,
                  indexes_heating_EM,
                  indexes_heating_heat,
                  double_res=False,
                  dz=None,
                  alpha = 0.0,
                  k_wg = None,
                  k_clad = None,
                  k_metal = None,
                  volume_constraint = False, 
                  heating_constraint = False,
                  continuation_scheme = False,
                  eliminate_excitation = False,
                  debug = False,
                  logfile=False):
        """
        Initialization of the main class.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ dVini: Initial value for the design variables.
        @ n_metal : Value for the refractive index of the metal.
        @ k_r : Value for the exctinction coefficient of the metal.
        @ n_wg: Value for the refractive index of the waveguide.
        @ n_clad: Value for the refractive index of the cladding.
        @ wl: Wavelength of the problem (Frequency domain solver).
        @ delta: Value of the effective refractive index (COMSOL).
        @ fR: Filtering radius.
        @ eta: parameter that controls threshold value.
        @ beta: parameter that controls the threshold sharpness.
        @ scaling: Length scale of the problem.
        @ vol_constraint_val: max. value of volume fraction in entire domain.
        @ indexes_design_region: indexes of elements in design region
        @ heat_constraint_val: max. value of volume fraction in heating region.
        @ indexes_heating: indexes of  elements in heating region
        @ volume_constraint: Boolean to activate volume constraint.
        @ heating_constraint: Boolean to activate heating constraint.
        """
        warnings.filterwarnings("ignore") # we eliminate all possible warnings to make the notebooks more readable.

        self.nElX_EM = nElx_EM
        self.nElY_EM = nEly_EM

        self.nElX_heat = nElx_heat
        self.nElY_heat = nEly_heat

        self.nElXcore = (nElx_heat)//2 + int(0.5*w_core) - (nElx_heat)//2 + int(0.5*w_core)
        self.nElYcore = (nEly_heat)//2 + int(0.5*h_core) - (nEly_heat)//2 + int(0.5*h_core)

        self.alpha = alpha # pamping factor
        self.n_metal = n_metal
        self.k_r = k_r
        self.n_wg = n_wg
        self.n_clad = n_clad
        self.mu = 1 # non-magnetic media
        self.wavelength = wl
        self.delta = delta
        self.deltaT = deltaT
        self.fR = fR
        self.dVini = dVini
        self.FOM_type = FOM_type

        self.double_res = double_res


        self.idx_wg_EM = indexes_wg_EM
        self.idx_wg_heat = indexes_wg_heat

        self.idxdr_EM = indexes_design_region_EM
        self.idxdr_heat = indexes_design_region_heat


        self.idxheat_EM = indexes_heating_EM
        self.idxheat_heat = indexes_heating_heat

        self.volume_constraint = volume_constraint
        self.vol_cons_val  =  vol_cons_val

        self.heating_constraint = heating_constraint
        self.heat_cons_val = heat_cons_val

        self.dis_0 = None
        self.dVs = None
        self.eta = eta
        self.beta = beta

        self.continuation_scheme = continuation_scheme

        self.eliminate_excitation = eliminate_excitation

        self.dz = dz
        self.k_wg = k_wg
        self.k_clad = k_clad
        self.k_metal = k_metal
        self.logfile = logfile


        # -----------------------------------------------------------------------------------
        # PHYSICS OF THE PROBLEM
        # ----------------------------------------------------------------------------------- 
        self.scaling = scaling # We give the scaling of the physical problem; i.e. 1e-9 for nm.

        self.phys = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.delta,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 

        self.physT = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.deltaT,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 

        #self.delta_n_wg = np.zeros_like((self.nElYcore, self.nElXcore))
        self.delta_n_wg = np.zeros((self.nElY_EM, self.nElX_EM, 4))
        self.constraint_3 = None

        # -----------------------------------------------------------------------------------
        # DISCRETIZATION OF THE PROBLEM
        # -----------------------------------------------------------------------------------                 
        self.dis_0 = dis(self.scaling,
                    self.nElX_EM,
                    self.nElY_EM,
                    #self.double_res,
                    debug)

        self.dis_heat = dis_heat(self.scaling,
                    self.nElX_heat,
                    self.nElY_heat,
                    #self.double_res,
                    debug)

        # We set the indexes of the discretization: i.e. system matrix, boundary conditions ...

        self.dis_0.index_set() 
        self.dis_heat.index_set() 

        # -----------------------------------------------------------------------------------  
        # FILTERING AND THRESHOLDING 
        # -----------------------------------------------------------------------------------   
        self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, self.beta) 
        self.filThr_heat =  filter_threshold(self.fR, self.nElX_heat, self.nElY_heat, self.eta, self.beta) 

        # -----------------------------------------------------------------------------------  
        # INITIALIZING DESIGN VARIABLES
        # -----------------------------------------------------------------------------------  
        self.dVs = self.dVini 

        if logfile:

            self.directory_opt, self.today = init_dir("_opt")
            self.directory_sweep, _ = init_dir("_sweep")

    def grayscale_constraint(self, x):

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFPST

            
            dFPST = calculate_design_field(x)
            N = len(np.array(dFPST[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]).flatten())
            constraint = 0.25 * np.sum((dFPST.flatten()*(1-dFPST.flatten()))) / N
            sens = 0.25 * (1-2*dFPST) / N

            sens_constraint = self.dis_0.filter_sens(self.filThr_EM, sens)[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]

            self.sens_grey = sens_constraint

            print("Greyscale constraint: ", np.real(constraint))

            return constraint.astype("float64"), sens_constraint.flatten()

    def geom_l_s(self,x):  # geometric lentghscale constraint for solid

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFP, dFPS, dFPST

            dFP, dFPS, dFPST = calculate_design_field(x)

            #dFPST = self.dis_0.dFPST
            #dFPS = self.dis_0.dFPS

            eta_e = self.eta_e
            n = self.num_el
            grad_dFPS = self.dis_0.d_dFPS
            grad_dFPS_2 = self.dis_0.d_dFPS
            d_thresh = self.dis_0.d_thresh
            d_thresh_filter = self.dis_0.d_thresh_filter
            d_filter =  self.dis_0.d_filter
            d_grad_filter = self.dis_0.d_grad_filter
            c = self.c

            self.g_s = g_s(dFPS, n, eta_e, c, grad_dFPS_2, dFPST)
            self.sens_g_s = sens_g_s(dFPS, n, eta_e, c, grad_dFPS_2, dFPST, grad_dFPS, d_thresh, d_thresh_filter, d_filter, d_grad_filter)[self.idxdr_EM[0,:], self.idxdr_EM[1,:]]

            return  self.g_s/self.eps_l - 1 , self.sens_g_s.flatten()/self.eps_l

    def geom_l_v(self,x):  # geometric lentghscale constraint for solid

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFP, dFPS, dFPST

            dFP, dFPS, dFPST = calculate_design_field(x)

            #dFPST = self.dis_0.dFPST
            #dFPS = self.dis_0.dFPS

            eta_d = self.eta_d
            n = self.num_el
            grad_dFPS = self.dis_0.d_dFPS
            grad_dFPS_2 = self.dis_0.d_dFPS
            d_thresh = self.dis_0.d_thresh
            d_thresh_filter = self.dis_0.d_thresh_filter
            d_filter =  self.dis_0.d_filter
            d_grad_filter = self.dis_0.d_grad_filter
            c = self.c

            self.g_v = g_v(dFPS, n, eta_d, c, grad_dFPS_2, dFPST)
            self.sens_g_v = sens_g_v(dFPS, n, eta_d, c, grad_dFPS_2, dFPST, grad_dFPS, d_thresh, d_thresh_filter, d_filter, d_grad_filter)[self.idxdr_EM[0,:], self.idxdr_EM[1,:]]

            return  self.g_v/self.eps_l - 1 , self.sens_g_v.flatten()/self.eps_l
    

    def solve_forward(self, dVs, solver, idx_RHS = None, val_RHS=None, eigval_num=0):
        """
        Function to solve the forward FEM problem in the frequency domain given a distribution of dielectric function in the simulation domain.
        """
        sens_heat = None
        E = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver, self.delta_n_wg, sens_heat, idx_RHS, val_RHS, eigval_num, self.eliminate_excitation)
        
        Ex = E [:self.dis_0.nodesX*self.dis_0.nodesY]  
        Ey = E [self.dis_0.nodesX*self.dis_0.nodesY:2*self.dis_0.nodesX*self.dis_0.nodesY]  
        Ez = E [2*self.dis_0.nodesX*self.dis_0.nodesY:]  
        FOM = self.plot_FOM(idx_RHS)
        self.E_unheated = np.real(self.dis_0.normE)
        #FOM = -1E9
        #print("FOM: ", -FOM)

        return Ex, Ey, Ez, FOM

    def solve_forward_T(self, dVs, solver, idx_RHS = None, val_RHS=None, eigval_num=0):
        """
        Function to solve the forward FEM problem in the frequency domain given a distribution of dielectric function in the simulation domain.
        """
        E = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.physT, self.filThr_EM, solver, self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, eigval_num, self.eliminate_excitation)
        
        Ex = E [:self.dis_0.nodesX*self.dis_0.nodesY]  
        Ey = E [self.dis_0.nodesX*self.dis_0.nodesY:2*self.dis_0.nodesX*self.dis_0.nodesY]  
        Ez = E [2*self.dis_0.nodesX*self.dis_0.nodesY:]  
        FOM = self.plot_FOM(idx_RHS)

        return Ex, Ey, Ez, FOM


    def solve_heat(self, dVs, solver, idx_RHS = None, val_RHS=None, eigval_num=0):
        "TBD"
        #T_field = self.dis_heat.FEM_sol_heat(dVs, self.FOM_type, self.idxdr_heat, self.idx_wg_heat,  self.physT, self.filThr_heat, solver, idx_RHS, val_RHS, eigval_num, self.eliminate_excitation)
        FOM, self.sens_heat = self.dis_heat.objective_grad(dVs, self.FOM_type, self.idxdr_heat, self.idx_wg_heat,  self.physT, self.filThr_heat, idx_RHS, val_RHS)
        #FOM  = self.dis_heat.compute_FOM(self.idx_wg_heat,  self.physT)
        #print("FOM / Average temperature in waveguide:  ", FOM)

        delta_n_delta_T_Si = 1.8E-4
        delta_n = delta_n_delta_T_Si * FOM

        #print("Simplified shift in refractive index: ", delta_n)

        self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

        self.plot_FOM_heat(idx_RHS)

        #self.deltaT = self.delta + 1.8E-4 * self.dis_heat.T_wg
        #print("New effective index: ", self.deltaT)

       #self.physT = phy(self.n_metal,
       #                 self.k_r,
       #                self.n_wg,
       #                 self.n_clad,
       #                 self.mu,
       #                 self.scaling,
       #                 self.wavelength,
       #                 self.deltaT,
       #                 self.dz,
       #                 k_wg= self.k_wg,
       #                 k_metal = self.k_metal,
       #                 k_clad = self.k_clad
       #                 ) 

        return self.dis_heat.T

    def solve_coupled(self, dVs, solver, idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None,  eigval_num=0):

        print("---------------------------------------------------")
        print("Solving for the unheated device ...")
        print("---------------------------------------------------")

        self.delta_n_wg = np.zeros_like(self.delta_n_wg) # we initialize the index change to zero

        # First, we solve the EM problem without any heat applied to it

        _, __, ___, self.FOM_1_EM = self.solve_forward(dVs, solver, idx_RHS_EM, val_RHS_EM, eigval_num=0)

        print("EM FOM, for unheated device: ", self.FOM_1_EM)

        # Then we solve the heat problem 

        print("---------------------------------------------------")
        print("Solving for the heat propagation...")
        print("---------------------------------------------------")
        _ = self.solve_heat(dVs, solver, idx_RHS_heat, val_RHS_heat)


        # This will give us the new refractive index in the waveguide

        print("---------------------------------------------------")
        print("Solving for the heated device ...")
        print("---------------------------------------------------")

        _, __, ___, self.FOM_2_EM = self.solve_forward_T(dVs, solver, idx_RHS_EM, val_RHS_EM, eigval_num=0)

        print("EM FOM, for heated device: ", self.FOM_2_EM)

    def optimize_coupled(self, maxItr, tol, algorithm,  idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None):

        #p = 4 # norm number
        #dVs_hom = np.zeros_like(self.dVini)
        np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
        self.z = 100000.0

        #self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
        
        #self.FOMini1, _ = self.dis_0.objective_grad(self.dVini, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
        #self.FOMini1, _ = FOM_sens_log(self.FOMini1, _)
        #self.FOMini1, _ = FOM_sens_division(self.FOMini1, _)

        #self.FOM_heat0, self.sens_heat = self.dis_heat.objective_grad(self.dVini, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.phys, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
        #self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)
        
        #self.FOMini2, _ = self.dis_0.objective_grad(self.dVini, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
        #self.FOMini2, _ = FOM_sens_log(self.FOMini2, _)
        #self.FOMini2, _ = FOM_sens_division(self.FOMini2, _)
        #print("With only cladding...")
        #self.delta_n_wg = np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
        #self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities

        

        self.maxItr = maxItr
        self.algorithm = algorithm
        self.tol = tol 

        def set_optimization_algorithm(algorithm, n):
            if algorithm == "MMA":
                opt = nlopt.opt(nlopt.LD_MMA, n)
            if algorithm == "BFGS":
                opt = nlopt.opt(nlopt.LD_LBFGS, n)
            return opt
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        global it_num, i_con # We define a global number of iterations to keep track of the step

        it_num = 0
        i_con = 0

        def f(x, grad): 

            global i_con
            global it_num

            if self.continuation_scheme:
                if (it_num+1) % 50 == 0:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    self.filThr_heat =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    i_con += 1

            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)

            # We calculate the FOM for the unheated design

            

            self.delta_n_wg = np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
            #self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.sens1 = - (1/self.FOM1**2) * self.sens1
            #FOM1 = 1. + 1 / self.FOM1 

            # We calculate the temperature FOM and the refractive index change 

            #FOM_heat, _ = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.phys, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            #FOM_heat += 1.0 # so we do not get problems with numerical accuracy
            #FOM_heat = 1. + 1 / FOM_heat 
            #self.sens_heat = - (1/FOM_heat**2) * self.sens_heat
            #self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

            # We calculate the FOM for the heated device

            #self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.sens2 = - (1/self.FOM2**2) * self.sens2
            #FOM2 = 1. + 1 / self.FOM2 

            # We put all of the FOMs together in a p-norm FOM and sensitivities
            #
            #FOM = p_norm(np.array([self.FOM1, self.FOM2]), p)
            #print("FOM total: ", FOM)
            #self.FOM_list [it_num] = FOM #/ self.FOM0

            #self.sens = sens_p_norm(np.array([self.FOM1, self.FOM2]), np.array([self.sens1, self.sens2]), p)

            #FOM = self.z
            #self.FOM_list [it_num] = FOM #/ self.FOM0

            
            if grad.size > 0:
                grad[:] = 0.0
            it_num += 1

            return self.z

        def minmax_FOM_1_plus_constraint(x, grad):

            #self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.phys, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            #self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)
            self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            self.FOM1, self.sens1 = FOM_sens_division(self.FOM1, self.sens1)
            self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_scaling(self.FOM1, self.sens1, 1/20000)
            #self.FOM1 = 1. + self.FOM1

            if grad.size > 0:
                grad[:] =  self.sens1.flatten() 

            print("FOM1: ", self.FOM1)

            self.constraint_1[it_num-1] = self.FOM1 #- 1

            #_ = self.solve_heat(x, "RHS", idx_RHS_heat, val_RHS_heat)
            #self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.phys, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            #self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)


            return self.FOM1 + self.z
        

        def minmax_FOM_2_plus_constraint(x, grad):

            self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

            self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            self.FOM2, self.sens2 = FOM_sens_division(self.FOM2, self.sens2)
            self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_scaling(self.FOM2, self.sens2, 1/20000)
            #self.FOM2 = 1. + self.FOM2 
            #plot_iteration(self.dis_0)

            print("FOM2: ", self.FOM2)

            if grad.size > 0:
                grad[:] =  self.sens2.flatten() 

            #plot_sens(self.dis_0, self.idxdr_EM, self.sens1)
            #raise()
            
            self.constraint_2[it_num-1] = self.FOM2 

            return self.FOM2 + self.z

        def grayscale_constraint(x,grad):

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFPST[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]

            
            dFPST = calculate_design_field(x)
            N = len(np.array(dFPST).flatten())
            constraint = 0.25 * np.sum((dFPST.flatten()*(1-dFPST.flatten()))) / N
            sens = 0.25 * (1-2*dFPST) / N

            sens_constraint = self.dis_0.filter_sens(self.filThr_EM, sens)[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]
            

            if grad.size > 0:
                grad[:] =  3E-2*sens_constraint.flatten() 

            self.constraint_3[it_num-1] = 3E-2*constraint

            print("Greyscale constraint: ", 3E-2*constraint)

            return 3E-2*constraint.astype("float64")

        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            betas = np.array([7.5, 10, 12.5, 15, 17.5, 20, 22.5])
        
                # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize
        opt = set_optimization_algorithm(algorithm, n)
        #opt.set_max_objective(f)
        opt.set_min_objective(f)
        opt.set_lower_bounds(LBdVs)
        opt.set_upper_bounds(UBdVs)
        opt.set_maxeval(maxItr)
        #opt.set_param("inner_maxeval", 5)
        #opt.set_ftol_rel(tol)

        opt.add_inequality_constraint(lambda x,grad: minmax_FOM_1_plus_constraint(x,grad), 1e-13)
        opt.add_inequality_constraint(lambda x,grad: minmax_FOM_2_plus_constraint(x,grad), 1e-13)
        #opt.add_inequality_constraint(lambda x,grad: grayscale_constraint(x,grad), 1e-1)

        self.FOM_list = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)
        self.constraint_1 = np.zeros(maxItr)
        self.constraint_2 = np.zeros(maxItr)
        self.constraint_3 = np.zeros(maxItr)

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs = opt.optimize(self.dVs) # we optimize for the design variables
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
        if self.logfile:
            create_logfile_optimization(self, idx_RHS_EM, val_RHS_EM, idx_RHS_heat, val_RHS_heat)

        return self.dVs

    def optimize_coupled_new(self, maxItr, tol, algorithm,  idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None):

       
        np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
        self.z = 10
        
        self.maxItr = maxItr
        self.algorithm = algorithm
        self.tol = tol 
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        global it_num, i_con # We define a global number of iterations to keep track of the step

        it_num = 0
        i_con = 0

        def f0(x): 

            global i_con
            global it_num

            if self.continuation_scheme:
                if (it_num+1) % 25 == 0:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    self.filThr_heat =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    i_con += 1

            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)

            self.delta_n_wg = np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
            self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)
            self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            p = 4 
            FOM = p_norm(np.array([self.FOM1, self.FOM2]), p)
            self.sens = sens_p_norm(np.array([self.FOM1, self.FOM2]), np.array([self.sens1, self.sens2]), p)
            self.FOM_list [it_num] = FOM
            #FOM, self.sens = FOM_sens_scaling(FOM, self.sens, 1/5E3)

            if self.logfile:

                fig, ax = plt.subplots(figsize=(14,10))
                extent = [-0.5*self.nElX_EM*self.scaling * 1e6, 0.5*self.nElX_EM*self.scaling * 1e6, -0.5*self.nElY_EM*self.scaling * 1e6, 0.5*self.nElY_EM*self.scaling * 1e6]
                wg_region = np.zeros((self.nElY_EM, self.nElX_EM))
                wg_region[self.idx_wg_EM[0,:], self.idx_wg_EM[1,:]] = 1

                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im = ax.imshow(np.real(self.dis_0.dFPST), cmap='binary', origin="lower", vmax=1, vmin=0, extent=extent)
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
                #print('Maximum real dielectric permittivity: ', np.max(np.real(dis.eps)))
                #print('Minimum real dielectric permittivity: ', np.min(np.real(dis.eps)))
                ax.set_xlabel('$X (\\mu m)$')
                ax.set_ylabel('$Y (\\mu m)$')
                import os
                directory = self.directory_opt+"/design_history"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig.savefig(self.directory_opt + "/design_history/design_it"+str(it_num)+".png")


            it_num += 1

            return -FOM, -self.sens[:, np.newaxis] #self.z, np.zeros_like(x) #gradient is zero! #FOM, self.sens[:, np.newaxis] #

        def minmax_FOM_unheated_constraint(x):

            self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_division(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_scaling(self.FOM1, self.sens1, 1/10)
            #self.FOM1 = 1. + self.FOM1

            print("FOM1: ", self.FOM1)

            self.constraint_1[it_num-1] = self.FOM1 

            return -self.FOM1+50-self.z, -self.sens1.flatten()
        

        def minmax_FOM_heated_constraint(x):

            self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

            self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_division(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_scaling(self.FOM2, self.sens2, 1/10)
            #self.FOM2 = 1. + self.FOM2 
            #plot_iteration(self.dis_0)

            print("FOM2: ", self.FOM2)

            self.constraint_2[it_num-1] = self.FOM2 

            return -self.FOM2+50-self.z, -self.sens2.flatten()

        def grayscale_constraint(x):


            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFPST

            
            dFPST = calculate_design_field(x)
            N = len(np.array(dFPST[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]).flatten())
            constraint = 0.25 * np.sum((dFPST.flatten()*(1-dFPST.flatten()))) / N
            sens = (0.25 * (1-2*dFPST) / N) 

            sens_constraint = self.dis_0.filter_sens(self.filThr_EM, sens)[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]

            #constraint, sens_constraint = FOM_sens_log(constraint, sens_constraint)

            self.sens_grey = sens_constraint 

            self.constraint_3[it_num-1] = np.real(constraint)

            print("Greyscale constraint: ", np.real(constraint))

            return constraint, self.sens_grey.flatten()

        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            betas = np.array([10, 15, 25, 50, 100, 100, 100])
        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize

        self.FOM_list = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)
        self.constraint_1 = np.zeros(maxItr)
        self.constraint_2 = np.zeros(maxItr)
        self.constraint_3 = np.zeros(maxItr)

        # -----------------------------------------------------------------------------------  
        # INITIALIZE OPTIMIZER
        # -----------------------------------------------------------------------------------
        m = 1 # number of constraint
        p = 0 #2 # number of objective functions in minmax
        #f = np.array([minmax_FOM_unheated_constraint, minmax_FOM_heated_constraint]) #, grayscale_constraint])
        f = np.array([])
        a0 = 1.0 # for minmax formulation 
        a = np.zeros(m)[:,np.newaxis] # p objective funcions and m constraints
        #a [0,0] = 1.0 
        #a [1:p,0] = 1.0 # for minmax formulation

        d = np.zeros(m)[:,np.newaxis]
        c = 1000 * np.ones(m)[:,np.newaxis]
        move = 0.01 # check this with Jonathan at some point: 0.2 for easy, 0.1 for hard problems.

        opt = optimizer(m, n, p, LBdVs[:,np.newaxis], UBdVs[:,np.newaxis], f0, f, a0, a, c, d, self.maxItr, move)

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs = opt.optimize(self.dVs[:,np.newaxis])
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
        if self.logfile:
            create_logfile_optimization(self, idx_RHS_EM, val_RHS_EM, idx_RHS_heat, val_RHS_heat)

        return self.dVs

    def optimize_coupled_minmax(self, maxItr, tol, algorithm,  idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None):

       
        np.zeros_like(self.delta_n_wg) # we initialize the index change to zero
        self.z = 10
        
        self.maxItr = maxItr
        self.algorithm = algorithm
        self.tol = tol 
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        #global it_num, i_con # We define a global number of iterations to keep track of the step
        global i_con, it_num
        global iteration_number_list

        #it_num = 0
        it_num = self.maxItr
        i_con = 0
        iteration_number_list = []

        def f0(x, it_num, plot=False): 

            global i_con
            #global it_num
            
            if self.continuation_scheme:
                if (it_num+1) % 50 == 0 and (it_num+1) not in  iteration_number_list:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    self.filThr_heat =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    self.alpha = alphas [i_con]
                    self.eps_l = eps_s[i_con]

                    self.phys = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.delta,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 

                    self.physT = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.deltaT,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 

                    i_con += 1

                    iteration_number_list.append(it_num+1)

            #print("----------------------------------------------")
            #print("Optimization iteration: ",it_num)

            self.delta_n_wg = np.zeros_like(self.delta_n_wg) # we initialize the index change to zero

            if self.logfile and plot:

                fig, ax = plt.subplots(2,2,figsize=(20,8))
                extent = [-0.5*self.nElX_EM*self.scaling * 1e6, 0.5*self.nElX_EM*self.scaling * 1e6, -0.5*self.nElY_EM*self.scaling * 1e6, 0.5*self.nElY_EM*self.scaling * 1e6]
                wg_region = np.zeros((self.nElY_EM, self.nElX_EM))
                wg_region[self.idx_wg_EM[0,:], self.idx_wg_EM[1,:]] = 1

                divider = make_axes_locatable(ax[0,1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                im0 = ax[0,0].imshow(np.real(self.dis_0.dFPST), cmap='binary', origin="lower", vmax=1, vmin=0, extent=extent)
                im =  ax[0,1].imshow(np.reshape(self.E_unheated, (self.dis_0.nodesY, self.dis_0.nodesX)), cmap='inferno', origin="lower", extent=extent)
                im00 = ax[1,0].imshow(np.real(self.dis_0.dFPST), cmap='binary', origin="lower", vmax=1, vmin=0, extent=extent)
                im =  ax[1,1].imshow(np.reshape(np.real(self.dis_0.normE), (self.dis_0.nodesY, self.dis_0.nodesX)), cmap='inferno', origin="lower", extent=extent)
                fig.colorbar(im0, cax=cax, orientation='vertical')
                #fig.colorbar(im0, cax=cax, orientation='vertical')
                ax[0,0].contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
                ax[0,1].contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
                ax[1,0].contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
                ax[1,1].contour(wg_region, levels=1, cmap='binary', linewidth=2, alpha=1, extent=extent)
                #print('Maximum real dielectric permittivity: ', np.max(np.real(dis.eps)))
                #print('Minimum real dielectric permittivity: ', np.min(np.real(dis.eps)))
                for axi in ax:
                    for axis in axi:
                        axis.set_xlabel('$X (\\mu m)$')
                        axis.set_ylabel('$Y (\\mu m)$')
                import os
                directory = self.directory_opt+"/design_history"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig.savefig(self.directory_opt + "/design_history/design_it"+str(it_num)+".png")

            it_num += 1

            return self.z, np.zeros_like(x) #gradient is zero! #FOM, self.sens[:, np.newaxis] #

        def minmax_FOM_unheated_constraint(x):

            self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM1, self.sens1 = FOM_sens_scaling(self.FOM1, self.sens1, 1E3)
            self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            self.E_unheated = np.real(self.dis_0.normE)
            #self.FOM1, self.sens1 = FOM_sens_division(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_log(self.FOM1, self.sens1)
            #self.FOM1, self.sens1 = FOM_sens_scaling(self.FOM1, self.sens1, 1/100)
            #self.FOM1 = 1. + self.FOM1

            print("FOM1: ", self.FOM1)

            #self.constraint_1[it_num-1] = self.FOM1 

            return -self.FOM1+50-self.z, -self.sens1 #-self.FOM1+50-self.z, -self.sens1.flatten()
        

        def minmax_FOM_heated_constraint(x):

            self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

            self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #self.FOM2, self.sens2 = FOM_sens_scaling(self.FOM2, self.sens2, 1E3)
            self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_division(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_log(self.FOM2, self.sens2)
            #self.FOM2, self.sens2 = FOM_sens_scaling(self.FOM2, self.sens2, 1/100)
            #self.FOM2 = 1. + self.FOM2 
            #plot_iteration(self.dis_0)

            print("FOM2: ", self.FOM2)

            #self.constraint_2[it_num-1] = self.FOM2 

            return -self.FOM2+50-self.z, -self.sens2 # -self.FOM2+50-self.z, -self.sens2.flatten()

        def grayscale_constraint(x):


            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFPST

            
            dFPST = calculate_design_field(x)
            N = len(np.array(dFPST[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]).flatten())
            constraint = 0.25 * np.sum((dFPST.flatten()*(1-dFPST.flatten()))) / N
            sens = (0.25 * (1-2*dFPST) / N) 

            sens_constraint = self.dis_0.filter_sens(self.filThr_EM, sens)[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]

            #constraint, sens_constraint = FOM_sens_log(constraint, sens_constraint)

            self.sens_grey = sens_constraint 

            #self.constraint_3[it_num-1] = np.real(constraint)

            print("Greyscale constraint: ", np.real(constraint))

            return constraint, self.sens_grey.flatten()

        def calculate_design_field(x):
            dFP = self.dis_0.material_distribution(x, self.idxdr_EM)
            dFPS = self.filThr_EM.density_filter(np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"), self.filThr_EM.filSca, dFP, np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"))
            dFPST = self.filThr_EM.threshold(dFPS)
            return dFPST[self.idxdr_EM[0,:], self.idxdr_EM[1,:]]

        def volume_constraint(x):
            dFPST = calculate_design_field(x)
            self.sens_vol =  np.ones_like(self.dVs)/len(dFPST.flatten())

            return (np.sum(dFPST.flatten())/len(dFPST.flatten()) - self.vol_cons_val).astype("float64"), self.sens_vol

        def minmax_FOM_unheated_constraint_no_sens(x):

            solver = "RHS"
            sens_heat = None
            eigval_num = 0

            E = self.dis_0.FEM_sol(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver, self.delta_n_wg, sens_heat, idx_RHS_EM, val_RHS_EM, eigval_num, self.eliminate_excitation)
            FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS_EM, self.phys)
            FOM = np.log10(FOM)

            return FOM +50-self.z

        def minmax_FOM_heated_constraint_no_sens(x):

            solver = "RHS"
            sens_heat = None
            eigval_num = 0

            self.FOM_heat, self.sens_heat = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)

            E = self.dis_0.FEM_sol(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.physT, self.filThr_EM, solver, self.delta_n_wg, self.dis_heat, idx_RHS_EM, val_RHS_EM, eigval_num, self.eliminate_excitation)
            FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS_EM, self.physT)
            FOM = np.log10(FOM)

            return FOM+50-self.z

        def volume_constraint_no_sens(x):

            dFPST = calculate_design_field(x)
            return (np.sum(dFPST.flatten())/len(dFPST.flatten()) - self.vol_cons_val).astype("float64")

        def volume_constraint_no_sens(x):

            dFPST = calculate_design_field(x)
            return (np.sum(dFPST.flatten())/len(dFPST.flatten()) - self.vol_cons_val).astype("float64")

        def geom_l_s_no_sens(x):

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFP, dFPS, dFPST

            dFP, dFPS, dFPST = calculate_design_field(x)

            eta_e = self.eta_e
            n = self.num_el
            grad_dFPS = self.dis_0.d_dFPS
            grad_dFPS_2 = self.dis_0.d_dFPS
            d_thresh = self.dis_0.d_thresh
            d_thresh_filter = self.dis_0.d_thresh_filter
            d_filter =  self.dis_0.d_filter
            d_grad_filter = self.dis_0.d_grad_filter
            c = self.c

            self.g_s = g_s(dFPS, n, eta_e, c, grad_dFPS_2, dFPST)
            return  self.g_s/self.eps_l - 1

        def geom_l_v_no_sens(x):

            def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFP, dFPS, dFPST

            dFP, dFPS, dFPST = calculate_design_field(x)

            #dFPST = self.dis_0.dFPST
            #dFPS = self.dis_0.dFPS

            eta_d = self.eta_d
            n = self.num_el
            grad_dFPS = self.dis_0.d_dFPS
            grad_dFPS_2 = self.dis_0.d_dFPS
            d_thresh = self.dis_0.d_thresh
            d_thresh_filter = self.dis_0.d_thresh_filter
            d_filter =  self.dis_0.d_filter
            d_grad_filter = self.dis_0.d_grad_filter
            c = self.c

            self.g_v = g_v(dFPS, n, eta_d, c, grad_dFPS_2, dFPST)

            return self.g_v/self.eps_l - 1

        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            factor = 1.5
            betas = self.beta * np.array([factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7])
            alphas = np.array([factor, factor**2, factor**3, factor**4, factor**5, factor**6, factor**7])
            eps_s = np.array([1E-4,1E-4, 1E-4, 1E-5, 1E-6, 1E-7, 1E-7, 1E-8])
        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize

        self.FOM_list = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)
        self.constraint_1 = np.zeros(maxItr)
        self.constraint_2 = np.zeros(maxItr)
        self.constraint_3 = np.zeros(maxItr)

        # LENGTHSCALE CONSTRAINT STUFF

        self.eta_e = 0.75 # eroded eta
        self.eta_d = 0.25 # dilated eta
        self.num_el = self.nElX_EM * self.nElY_EM
        self.c = 16000 # as discussed with Goktug (Rasmus?) / r^4
        lw = 2 * self.fR * self.scaling # this should be the imposed lengthscale
        self.eps_l = 1E-6 # relaxing lengthscale constraint

        # -----------------------------------------------------------------------------------  
        # INITIALIZE OPTIMIZER
        # -----------------------------------------------------------------------------------
        m = 3 # number of constraint
        p = 2 #2 # number of objective functions in minmax
        f = np.array([minmax_FOM_unheated_constraint, minmax_FOM_heated_constraint, volume_constraint]) #, grayscale_constraint])
        f_no_d =  np.array([minmax_FOM_unheated_constraint_no_sens, minmax_FOM_heated_constraint_no_sens, volume_constraint_no_sens])# function call without derivative
        #f = np.array([])
        a0 = 1.0 # for minmax formulation 
        a = np.zeros(m)[:,np.newaxis] # p objective funcions and m constraints
        a [0,0] = 1.0 
        a [1:p,0] = 1.0 # for minmax formulation

        d = np.zeros(m)[:,np.newaxis]
        c = 1000 * np.ones(m)[:,np.newaxis]

        m_c = 5
        f_c = np.array([minmax_FOM_unheated_constraint, minmax_FOM_heated_constraint, volume_constraint, self.geom_l_s, self.geom_l_v]) #continuation with lengthscale constraint
        f_no_c =  np.array([minmax_FOM_unheated_constraint_no_sens, minmax_FOM_heated_constraint_no_sens, volume_constraint_no_sens, geom_l_s_no_sens, geom_l_v_no_sens])# function call without derivative
        a_c = np.zeros(m_c)[:,np.newaxis] # p objective funcions and m constraints
        a_c [0,0] = 1.0 
        a_c [1:p,0] = 1.0 # for minmax formulation
        d_c = np.zeros(m_c)[:,np.newaxis]
        c_c = 1000 * np.ones(m_c)[:,np.newaxis]
        #c_c [3:,:] = 1E4 # Try to make the lentghscale constraint more dominant
        maxiter_cont_l = 150

        move = 0.1 # check this with Jonathan at some point: 0.2 for easy, 0.1 for hard problems.

        self.opt = optimizer(m, n, p, LBdVs[:,np.newaxis], UBdVs[:,np.newaxis], f0, f, f_no_d, f_c, f_no_c, maxiter_cont_l, a0, a, c, d, a_c, c_c, d_c, self.maxItr, move, type_MMA="GCMMA", maxiniter=1, logfile=True, directory=self.directory_opt)
        #self.opt = optimizer(m, n, p, LBdVs[:,np.newaxis], UBdVs[:,np.newaxis], f0, f, a0, a, c, d, self.maxItr, move, type_MMA="MMA")

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs, self.FOM_list, self.constraint_1, self.constraint_2, self.constraint_3, self.constraint_4, self.constraint_5 = self.opt.optimize(self.dVs[:,np.newaxis])
        #self.dVhist = self.opt.dVhist 
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
        if self.logfile:
            create_logfile_optimization(self, idx_RHS_EM, val_RHS_EM, idx_RHS_heat, val_RHS_heat)

        return self.dVs

    
    def optimize(self, maxItr, tol, algorithm, idx_RHS , val_RHS):
        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ algorithm: Algorithm to be used by the optimizer i.e. MMA, BFGS. 
        """
        
        p = 4 # norm number
        dVs_hom = np.zeros_like(self.dVini)
        self.delta_n_wg = np.zeros((self.nElYcore, self.nElXcore)) # we initialize the index change to zero
        self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities
        self.z = - self.FOM0 #self.FOM0
        self.x_old = self.dVini


        #self.FOM_0 = p_norm(np.array([FOM0, FOM0]), p)

        self.maxItr = maxItr

        def set_optimization_algorithm(algorithm, n):
            if algorithm == "MMA":
                opt = nlopt.opt(nlopt.LD_MMA, n)
            if algorithm == "BFGS":
                opt = nlopt.opt(nlopt.LD_LBFGS, n)
            return opt
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        global it_num, i_con # We define a global number of iterations to keep track of the step

        it_num = 0
        i_con = 0

        def f(x, grad): 

            global i_con
            global it_num

            if self.continuation_scheme:
                if (it_num+1) % 100 == 0:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    i_con += 1

            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)
            self.FOM, self.sens = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS , val_RHS, self.eliminate_excitation)

            #self.sens1 = - (1/self.FOM1**2) * self.sens1
            #FOM1 = 1. + 1 / self.FOM1 #self.z
            #self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, idx_RHS , val_RHS, self.eliminate_excitation)
            #self.sens2 = - (1/self.FOM2**2) * self.sens2
            #FOM2 = 1. + 1 / self.FOM2 #self.z
            #FOM = self.z
            #FOM = -p_norm(np.array([self.FOM1, self.FOM2]), p)
            #print("FOM: ", -FOM)
            self.FOM_list [it_num] = -self.FOM #FOM-1 #/ self.FOM0
            #self.sens = sens_p_norm(np.array([self.FOM1, self.FOM2]), np.array([self.sens1, self.sens2]), p)
            
            if grad.size > 0:
                grad[:] = -self.sens #np.zeros_like(self.dVs.flatten()) 

            it_num += 1

            return -self.FOM

        def calculate_design_field(x):
            dFP = self.dis_0.material_distribution(x, self.idxdr_EM)
            dFPS = self.filThr_EM.density_filter(np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"), self.filThr_EM.filSca, dFP, np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"))
            dFPST = self.filThr_EM.threshold(dFPS)
            return dFPST[self.idxdr_EM[0,:], self.idxdr_EM[1,:]]

        def volume_constraint(x, grad, vol_max):
            dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad[:] =  np.ones_like(self.dVs())

            return ( np.sum(dFPST.flatten()) - vol_max).astype("float64")

        def heating_constraint(x, grad, indexes_heating,  vol_max):
            global it_num

            dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad [:] = 0.0
                #grad[indexes_heating] = - np.ones_like(indexes_heating) / vol_max
            
            self.heat_const_list[it_num-1] = np.real(np.mean(self.dis_0.dFPST[indexes_heating[0,:],indexes_heating[1,:].flatten()]))
            print( np.real(np.mean(self.dis_0.dFPST[indexes_heating[0,:],indexes_heating[1,:]].flatten())))
            #print(- np.real(np.sum(self.dis_0.dFPST.flatten()[indexes_heating])/ (len(indexes_heating)*vol_max)))

            #it_num += 1

            return (- np.real(np.mean(self.dis_0.dFPST[indexes_heating[0,:],indexes_heating[1,:]].flatten())/ (vol_max)) + 1).astype("float64")

        def minmax_FOM_1_plus_constraint(x, grad):

            self.FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, idx_RHS , val_RHS, self.eliminate_excitation)
            self.sens1 = - (1/self.FOM1**2) * self.sens1
            self.FOM1 = 1. + 1 / self.FOM1 
            if grad.size > 0:
                #print(self.sens1.flatten())
                grad[:] =  self.sens1.flatten() 

            #print('test 0: ', self.FOM1-self.z)
            #print('test 1: ', -FOM1/self.z - 1)

            #print(self.FOM1 - self.z)
            print("FOM: ", self.FOM)

            #print(self.sens1.flatten())
            #print(self.sens2.flatten())

            #raise()

            self.constraint_1[it_num-1] = self.FOM - 1

            return self.FOM1  #- self.z

        def minmax_FOM_1_minus_constraint(x, grad):

            #FOM1, self.sens1 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr, self.idx_wg, self.phys, self.filThr, idx_RHS , val_RHS, self.eliminate_excitation)

            if grad.size > 0:
                grad[:] =  -self.sens1.flatten() 

            return -self.FOM1 - self.z
        def minmax_FOM_2_plus_constraint(x, grad):

            self.FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT, self.filThr_EM, self.delta_n_wg, idx_RHS , val_RHS, self.eliminate_excitation)
            self.sens2 = - (1/self.FOM2**2) * self.sens2
            self.FOM2 = 1. + 1 / self.FOM2 
            #print(self.FOM2 - self.z)
            #print('test 2: ', FOM2/self.z - 1)
            #print('test 3: ', -FOM2/self.z - 1)
            #raise()
            print("FOM2: ", self.FOM2)

            if grad.size > 0:
                grad[:] =  self.sens2.flatten() 
            
            self.constraint_2[it_num-1] = self.FOM2 - 1

            return self.FOM2 #- self.z

        def minmax_FOM_2_minus_constraint(x, grad):

            #FOM2, self.sens2 = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr, self.idx_wg, self.physT, self.filThr, idx_RHS , val_RHS, self.eliminate_excitation)

            if grad.size > 0:
                grad[:] =  -self.sens2.flatten() / self.z

            return -self.FOM2 - self.z


        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            betas = np.array([2., 5., 10., 25., 50.])
        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize
        opt = set_optimization_algorithm(algorithm, n)
        opt.set_min_objective(f)
        opt.set_lower_bounds(LBdVs)
        opt.set_upper_bounds(UBdVs)
        opt.set_maxeval(maxItr)
        opt.set_ftol_rel(tol)


        self.FOM_list = np.zeros(maxItr)
        self.constraint_1 = np.zeros(maxItr)
        self.constraint_2 = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)

        #if self.volume_constraint:
        #    vol_max = len(self.dVs) * self.vol_cons_val
        #    opt.add_inequality_constraint(lambda x,grad: volume_constraint(x,grad, vol_max), 1e-8) 

        if self.heating_constraint:
            vol_heat_max = self.heat_cons_val 
            opt.add_inequality_constraint(lambda x,grad: heating_constraint(x,grad, self.idxheat_EM, vol_heat_max), 1e-8) 

        #opt.add_inequality_constraint(lambda x,grad: minmax_FOM_1_plus_constraint(x,grad), 1e-13)
        #opt.add_inequality_constraint(lambda x,grad: minmax_FOM_2_plus_constraint(x,grad), 1e-13)
        #opt.add_inequality_constraint(lambda x,grad: minmax_FOM_1_minus_constraint(x,grad), 1e-8)
        #opt.add_inequality_constraint(lambda x,grad: minmax_FOM_2_minus_constraint(x,grad), 1e-8)

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs = opt.optimize(self.dVs) # we optimize for the design variables
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
       
        # -----------------------------------------------------------------------------------  
        # FINAL BINARIZED DESIGN EVALUATION
        # -----------------------------------------------------------------------------------

        #print("----------------------------------------------")
        #print("Final binarized design evaluation")
        #print("----------------------------------------------")
        #beta_old = self.filThr.beta
        #self.filThr.beta = 1000 # we set a very high treshold to achieve binarization
        #FOM, self.sens = self.dis_0.objective_grad(self.dVs, self.FOM_type, self.idxdr, self.idx_wg, self.phys,self.filThr, idx_RHS , val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities
        
        #self.dVs = self.dis_0.material_distribution(self.dVs, self.idxdr)
        #self.dVs = self.filThr.density_filter(np.ones((self.nElY, self.nElX), dtype="complex128"), self.filThr.filSca, self.dVs, np.ones((self.nElY, self.nElX), dtype="complex128"))
        #self.dVs = self.filThr.threshold(self.dVs)[self.idxdr[0,:], self.idxdr[1,:]].flatten()
        
        #self.filThr.beta = beta_old

        return self.dVs

    def optimize_kill(self, maxItr, tol, algorithm, idx_RHS , val_RHS):
        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ algorithm: Algorithm to be used by the optimizer i.e. MMA, BFGS. 
        """

        self.maxItr = maxItr

        def set_optimization_algorithm(algorithm, n):
            if algorithm == "MMA":
                opt = nlopt.opt(nlopt.LD_MMA, n)
            if algorithm == "BFGS":
                opt = nlopt.opt(nlopt.LD_LBFGS, n)
            return opt
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        global it_num, i_con # We define a global number of iterations to keep track of the step

        it_num = 0
        i_con = 0

        def f(x, grad):

            global sens
            global i_con
            global it_num

            if self.continuation_scheme:
                if (it_num+1) % 10 == 0:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr_EM =  filter_threshold(self.fR, self.nElX_EM, self.nElY_EM, self.eta, betas[i_con]) 
                    i_con += 1

            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)
            FOM, sens = self.dis_0.objective_grad(x, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, self.delta_n_wg, idx_RHS , val_RHS, self.eliminate_excitation)
            self.FOM_list [it_num] =FOM
            
            if grad.size > 0:
                grad[:] = sens.flatten()

            it_num += 1

            return FOM

        def calculate_design_field(x):
            dFP = self.dis_0.material_distribution(x, self.idxdr_EM)
            dFPS = self.filThr_EM.density_filter(np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"), self.filThr_EM.filSca, dFP, np.ones((self.nElY_EM, self.nElX_EM), dtype="complex128"))
            dFPST = self.filThr_EM.threshold(dFPS)
            return dFPST

        def volume_constraint(x, grad, vol_max):
            global it_num
            dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad[:] =  sens.flatten()
            self.heat_const_list[it_num] = np.sum(dFPST.flatten()) / len(dFPST.flatten())
            it_num += 1
            return ( np.sum(dFPST.flatten()) - vol_max).astype("float64")

        def heating_constraint(x, grad, indexes_heating,  vol_max):
            global it_num

            dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad[indexes_heating] =  sens.flatten()[indexes_heating]
            
            
            self.heat_const_list[it_num] = np.sum(dFPST.flatten()[indexes_heating]) / len(self.idxheat_EM)
            it_num += 1

            return ( np.sum(dFPST.flatten()[indexes_heating])- vol_max).astype("float64")

        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            betas = np.array([5., 25., 50., 100.])
        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize
        opt = set_optimization_algorithm(algorithm, n)
        opt.set_min_objective(f)
        opt.set_lower_bounds(LBdVs)
        opt.set_upper_bounds(UBdVs)
        opt.set_maxeval(maxItr)
        #opt.set_ftol_rel(tol)


        self.FOM_list = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)

        if self.volume_constraint:
            vol_max = len(self.dVs) * self.vol_cons_val
            opt.add_inequality_constraint(lambda x,grad: volume_constraint(x,grad, vol_max), 1e-8) 

        if self.heating_constraint:
            vol_heat_max = self.heat_cons_val * len(self.idxheat_EM)
            opt.add_inequality_constraint(lambda x,grad: heating_constraint(x,grad, self.idxheat_EM, vol_heat_max), 1e-8) 

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs = opt.optimize(self.dVs) # we optimize for the design variables
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")

        # -----------------------------------------------------------------------------------  
        # FINAL BINARIZED DESIGN EVALUATION
        # -----------------------------------------------------------------------------------

        #print("----------------------------------------------")
        #print("Final binarized design evaluation")
        #print("----------------------------------------------")
        #beta_old = self.filThr.beta
        #self.filThr.beta = 1000 # we set a very high treshold to achieve binarization
        #FOM, self.sens = self.dis_0.objective_grad(self.dVs, self.FOM_type, self.idxdr, self.idx_wg, self.phys,self.filThr, idx_RHS , val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities
        
        #self.dVs = self.dis_0.material_distribution(self.dVs, self.idxdr)
        #self.dVs = self.filThr.density_filter(np.ones((self.nElY, self.nElX), dtype="complex128"), self.filThr.filSca, self.dVs, np.ones((self.nElY, self.nElX), dtype="complex128"))
        #self.dVs = self.filThr.threshold(self.dVs)[self.idxdr[0,:], self.idxdr[1,:]].flatten()
        
        #self.filThr.beta = beta_old

        return self.dVs

    def optimize_heat(self, maxItr, tol, algorithm, idx_RHS , val_RHS):
        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ maxItr: Maximum number of iterations of the optimizer. 
        @ algorithm: Algorithm to be used by the optimizer i.e. MMA, BFGS. 
        """

        self.maxItr = maxItr

        def set_optimization_algorithm(algorithm, n):
            if algorithm == "MMA":
                opt = nlopt.opt(nlopt.LD_MMA, n)
            if algorithm == "BFGS":
                opt = nlopt.opt(nlopt.LD_LBFGS, n)
            return opt
        
        # -----------------------------------------------------------------------------------  
        
        LBdVs = np.zeros(len(self.dVs)) # Lower bound on design variables
        UBdVs = np.ones(len(self.dVs)) # Upper bound on design variables

        # -----------------------------------------------------------------------------------  
        # FUNCTION TO OPTIMIZE AS USED BY NLOPT
        # ----------------------------------------------------------------------------------- 
        global it_num, i_con # We define a global number of iterations to keep track of the step

        it_num = 0
        i_con = 0

        def f(x, grad):

            global i_con,  it_num

            if self.continuation_scheme:
                if (it_num+1) % 30 == 0:
                    print("NEW BETA: ", betas[i_con])
                    self.beta =  betas[i_con]
                    self.filThr =  filter_threshold(self.fR, self.nElX_heat, self.nElY_heat, self.eta, betas[i_con]) 
                    i_con += 1

            print("----------------------------------------------")
            print("Optimization iteration: ",it_num)
            FOM, self.sens = self.dis_heat.objective_grad(x, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.phys, self.filThr_heat, idx_RHS , val_RHS, self.eliminate_excitation)
            print("FOM: ", FOM)
            self.FOM_list [it_num] = FOM
            
            if grad.size > 0:
                grad[:] = self.sens.flatten()

            it_num += 1
            
            return FOM

        def calculate_design_field(x):
            dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
            dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
            dFPST = self.filThr_heat.threshold(dFPS)
            return dFPST[self.idxdr_heat[0,:], self.idxdr_heat[1,:]]

        def volume_constraint(x, grad, vol_max):
            #dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad[:] =  self.sens.flatten()

            self.heat_const_list[it_num-1] = np.real(np.sum((self.dis_heat.dFPST[self.idxdr[0, :], self.idxdr_heat[1,:]]).flatten())) / len(self.idxdr_heat[0,:])
            print(np.real(np.sum((self.dis_heat.dFPST[self.idxdr_heat[0, :], self.idxdr_heat[1,:]]).flatten())) / len(self.idxdr_heat[0,:]))

            return ( np.real(np.sum((self.dis_heat.dFPST[self.idxdr_heat[0, :], self.idxdr_heat[1,:]]).flatten())/ len(self.idxdr_heat[0,:])) - vol_max).astype("float64")

        def heating_constraint(x, grad, indexes_heating,  vol_max):
            global it_num

            dFPST = calculate_design_field(x)
            if grad.size > 0:
                grad[indexes_heating] = - self.sens.flatten()[indexes_heating]
            
           
            self.heat_const_list[it_num] = np.mean(dFPST.flatten()[indexes_heating]) 
            #it_num += 1

            return (- np.mean(dFPST.flatten()[indexes_heating]) + vol_max).astype("float64")

        # -----------------------------------------------------------------------------------  
        # CONTINUATION SCHEME
        # -----------------------------------------------------------------------------------

        if self.continuation_scheme:
            
            betas = np.array([2., 5., 10., 25., 50.])
        
        # -----------------------------------------------------------------------------------  
        # OPTIMIZATION PARAMETERS
        # -----------------------------------------------------------------------------------
        n = len(self.dVs) # number of parameters to optimize
        opt = set_optimization_algorithm(algorithm, n)
        opt.set_min_objective(f)
        opt.set_lower_bounds(LBdVs)
        opt.set_upper_bounds(UBdVs)
        opt.set_maxeval(maxItr)
        opt.set_ftol_rel(tol)


        self.FOM_list = np.zeros(maxItr)
        self.heat_const_list = np.zeros(maxItr)

        if self.volume_constraint:
            opt.add_inequality_constraint(lambda x,grad: volume_constraint(x,grad, self.vol_cons_val), 1e-8) 

        if self.heating_constraint:
            opt.add_inequality_constraint(lambda x,grad: heating_constraint(x,grad, self.idxheat, self.heat_cons_val), 1e-8) 

        # -----------------------------------------------------------------------------------  
        # RUN OPTIMIZATION
        # -----------------------------------------------------------------------------------
        start = time.time() # we track the total optimization time
        self.dVs = opt.optimize(self.dVs) # we optimize for the design variables
        end =  time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("----------------------------------------------")
        print("Total optimization time: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")

        # -----------------------------------------------------------------------------------  
        # FINAL BINARIZED DESIGN EVALUATION
        # -----------------------------------------------------------------------------------

        #print("----------------------------------------------")
        #print("Final binarized design evaluation")
        #print("----------------------------------------------")
        #beta_old = self.filThr.beta
        #self.filThr.beta = 1000 # we set a very high treshold to achieve binarization
        #FOM, self.sens = self.dis_heat.objective_grad(self.dVs, self.FOM_type, self.idxdr, self.idx_wg, self.phys, self.filThr, idx_RHS , val_RHS, self.eliminate_excitation)
        
        #self.dVs = self.dis_0.material_distribution(self.dVs, self.idxdr)
        #self.dVs = self.filThr.density_filter(np.ones((self.nElY, self.nElX), dtype="complex128"), self.filThr.filSca, self.dVs, np.ones((self.nElY, self.nElX), dtype="complex128"))
        #self.dVs = self.filThr.threshold(self.dVs)[self.idxdr[0,:], self.idxdr[1,:]].flatten()
        
        #self.filThr.beta = beta_old

        return self.dVs


    def iteration_history(self, minmax=False, save=False, dir=None):

        print("----------------------------------------------")
        print("Iteration history")
        print("----------------------------------------------")
        if minmax:
            plot_it_history_minmax(self.maxItr, self.FOM_list, self.constraint_1, self.constraint_2, self.constraint_3, it_num, save, dir)
            plot_lam_minmax(self.maxItr, self.opt, it_num, save, dir)
            plot_geom_minmax(self.maxItr, self.opt, it_num, save, dir)
        else: 
            plot_it_history(self.maxItr, self.FOM_list, self.heat_const_list, self.heat_cons_val, it_num)

    def change_dVs(self, iteration):

        def calculate_design_field(x):
                dFP = self.dis_heat.material_distribution(x, self.idxdr_heat)
                dFPS = self.filThr_heat.density_filter(np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"), self.filThr_heat.filSca, dFP, np.ones((self.nElY_heat, self.nElX_heat), dtype="complex128"))
                dFPST = self.filThr_heat.threshold(dFPS)
                return dFPST

        print("----------------------------------------------")
        print("Change in design variables in iteration "+str(iteration)+" :")
        print("----------------------------------------------")
        delta_dV = np.abs( self.dVhist[:, iteration] - self.dVhist[:, iteration-1] )
        delta_dV_real = calculate_design_field(delta_dV)
        plot_change_dVs (delta_dV, delta_dV_real, self.dis_0, self.idxdr_heat)


    def plot_FOM(self, idx_RHS):
        """
        Function to _ka the Ez
        """
        if self.FOM_type == "linear":
            FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS, self.phys)

        elif self.FOM_type == "log":
            _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS, self.phys)
        #print("FOM: ", FOM)
        plot_Enorm(self.dis_0)
        plot_Ex(self.dis_0)
        plot_Ex_im(self.dis_0)
        plot_Ey(self.dis_0)
        plot_Ez(self.dis_0)
        #plot_Hz(self.dis_0)
        #plot_Hx(self.dis_0)
        #plot_Hy(self.dis_0)

        return FOM

    def plot_FOM_heat(self, idx_RHS):
        """
        Function to plot the T
        """
        
        plot_T(self.dis_heat)


    
    def sens_check (self, dVs, idx_RHS, val_RHS, delta_dV=1e-4):
        """
        Sensitivity check using finite-differences
        """ 
        sens = np.zeros(len(dVs))
        #self.EzHz = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.phys, self.filThr_EM, "RHS", self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, eigval_num=None)
        #FOM0, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
        FOM0, _ = self.dis_0.objective_grad(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS , val_RHS, self.eliminate_excitation)

        #self.EzHzT = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.physT, self.filThr_EM, "RHS", self.delta_n_wg, idx_RHS, val_RHS, eigval_num=None)
        #FOM2, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

        #p = 2 # norm number
        #FOM0 = -p_norm(np.array([FOM1, FOM2]), p)

        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            #self.EzHz = self.dis_0.FEM_sol(dVs_new, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.phys, self.filThr_EM, "RHS", self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, eigval_num=None)
            #FOM_new, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            #self.EzHzT = self.dis_0.FEM_sol(dVs_new, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.physT, self.filThr_EM, "RHS", self.delta_n_wg, idx_RHS, val_RHS, eigval_num=None)
            #FOM2, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_new, _ = self.dis_0.objective_grad(dVs_new, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, self.eliminate_excitation)
            #p = 2 # norm number
            #FOM_new = -p_norm(np.array([FOM1, FOM2]), p)
            sens [i] = (FOM_new - FOM0) / delta_dV
            dVs_new [i] -= delta_dV

        plot_sens(self.dis_0, self.idxdr_EM, sens)

        return sens

    def sens_check_heat (self, dVs, idx_RHS, val_RHS, delta_dV=1e-4):
        """
        Sensitivity check using finite-differences
        """ 
        sens = np.zeros(len(dVs))
        #self.T = self.dis_heat.FEM_sol_heat(dVs, self.FOM_type, self.idxdr_heat, self.idx_wg_heat,  self.physT, self.filThr_heat, "RHS", idx_RHS, val_RHS, eigval_num=None)
        #FOM_0 = self.dis_heat.compute_FOM(self.idx_wg_heat,self.physT)
        FOM_0, _= self.dis_heat.objective_grad(dVs, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS, val_RHS, self.eliminate_excitation)
        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            #self.T = self.dis_heat.FEM_sol_heat(dVs_new, self.FOM_type, self.idxdr_heat, self.idx_wg_heat,  self.physT, self.filThr_heat, "RHS", idx_RHS, val_RHS, eigval_num=None)
            #FOM_new = self.dis_heat.compute_FOM(self.idx_wg_heat, self.physT)
            FOM_new, _= self.dis_heat.objective_grad(dVs_new, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS, val_RHS, self.eliminate_excitation)
            sens [i] = (FOM_new - FOM_0) / delta_dV
            dVs_new [i] -= delta_dV

        plot_sens(self.dis_heat, self.idxdr_heat, -sens)

        return -sens

    def sens_check_coupled (self, dVs, idx_RHS_EM, val_RHS_EM, idx_RHS_heat, val_RHS_heat, delta_dV=1e-4):
        """
        Sensitivity check using finite-differences
        """ 
        sens = np.zeros(len(dVs))
        #self.delta_n_wg = np.zeros((self.nElYcore, self.nElXcore))
        #self.solve_coupled(dVs, solver='RHS', idx_RHS_EM = idx_RHS_EM, val_RHS_EM=val_RHS_EM, idx_RHS_heat = idx_RHS_heat, val_RHS_heat=val_RHS_heat,  eigval_num=0)
        #FOM_0 = self.FOM_2_EM
        #_, __ = self.dis_0.objective_grad(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
        _, self.sens_heat = self.dis_heat.objective_grad(dVs, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
        self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)
        FOM_0, _ = self.dis_0.objective_grad(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
        #eps_0 = self.dis_0.eps[self.idxdr_EM[0,:], self.idxdr_EM[1,:]].flatten()
        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            #self.delta_n_wg = np.zeros((self.nElYcore, self.nElXcore))
            #self.solve_coupled(dVs, solver='RHS', idx_RHS_EM = idx_RHS_EM, val_RHS_EM=val_RHS_EM, idx_RHS_heat = idx_RHS_heat, val_RHS_heat=val_RHS_heat,  eigval_num=0)
            #FOM_0 = self.FOM_2_EM
            #_, __ = self.dis_0.objective_grad(dVs_new, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation) # we compute the FOM and sensitivities
            _, self.sens_heat = self.dis_heat.objective_grad(dVs_new, self.FOM_type, self.idxdr_heat, self.idx_wg_heat, self.physT, self.filThr_heat, idx_RHS_heat , val_RHS_heat, self.eliminate_excitation)
            self.delta_n_wg = self.calculate_index_change(self.dis_heat.T, self.idx_wg_heat)
            FOM_new, _ = self.dis_0.objective_grad(dVs_new, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.physT,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS_EM , val_RHS_EM, self.eliminate_excitation)
            #eps_new = self.dis_0.eps[self.idxdr_EM[0,:], self.idxdr_EM[1,:]].flatten() 
            sens [i] = (np.log10(FOM_new)-np.log10(FOM_0)) / delta_dV
            #sens [i] = (eps_new [i] - eps_0 [i]) / delta_dV
            dVs_new [i] -= delta_dV

        plot_sens(self.dis_0, self.idxdr_heat, sens)

        return sens

    def sens_check_grayscale (self, dVs, delta_dV=1e-4):
        """
        Sensitivity check using finite-differences
        """ 
        sens = np.zeros(len(dVs))
        val0 = self.grayscale_constraint(dVs)[0]
        dVs_new = dVs

        for i in range(len(dVs)):
            dVs_new [i] += delta_dV
            val_new = self.grayscale_constraint(dVs_new)[0]
            sens [i] = (val_new - val0) / delta_dV
            #sens [i] = (eps_new [i] - eps_0 [i]) / delta_dV
            dVs_new [i] -= delta_dV

        plot_sens(self.dis_0, self.idxdr_heat, sens)

        return sens

    
    def plot_material_interpolation(self, save=False, dir=None):
        """
        Function to plot the material interpolation after a given simulation.
        """
        try:
            plot_mi(self.dis_heat,  self.idx_wg_heat, save, dir)
        except:    
            plot_mi(self.dis_0,  self.idx_wg_EM, save, dir)
        #plot_mi_imag(self.dis_0)

    def plot_permittivity(self):
        plot_perm(self.dis_0,  self.idx_wg_EM)
        plot_perm_wg(self.dis_0,  self.idx_wg_EM)

    def plot_sensitivities(self):
        """
        Function to plot the sensitivities after a given simulation.
        """
        
        plot_sens(self.dis_0, self.idxdr_EM, self.sens)

        return self.sens

    def plot_sensitivities_heat(self):
        """
        Function to plot the sensitivities after a given simulation.
        """
        
        plot_sens(self.dis_heat, self.idxdr_heat, self.sens)

        return self.sens

    def plot_sensitivities_coupled(self):
        """
        Function to plot the sensitivities after a given simulation.
        """
        
        plot_sens(self.dis_0, self.idxdr_heat, self.sens2) # CHECK THIS

        return self.sens2

    def plot_sensitivities_grayscale(self):
        """
        Function to plot the sensitivities after a given simulation.
        """
        
        plot_sens(self.dis_0, self.idxdr_heat, self.sens_grey) # CHECK THIS

        return self.sens_grey

    def plot_sensitivities_coupled_eps(self):
        """
        Function to plot the sensitivities after a given simulation.
        """
        
        plot_sens(self.dis_0, self.idxdr_heat, self.dis_0.depsdx[self.idxdr_EM[0,:], self.idxdr_EM[1,:]].flatten()) 
    

        return self.dis_0.depsdx[self.idxdr_EM[0,:], self.idxdr_EM[1,:]].flatten()

    def prop_const_sweep_peak(self, neffmin, neffmax, neff0, neffT, N, dVs, idx_RHS, val_RHS):


        self.phys = phy(self.n_metal,
                            self.k_r,
                            self.n_wg,
                            self.n_clad,
                            self.mu,
                            self.scaling,
                            self.wavelength,
                            self.delta) 
        
        dVs_hom = np.zeros_like(self.dVini)
        #self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, idx_RHS, val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities

        neff_array = np.linspace(neffmin, neffmax, N)
        FOM_list = []

        print("---------------------------------------------------")
        print("Calculating the FOM for different propagation constants...")
        print("---------------------------------------------------")
        for i in tqdm(range(len(neff_array))):

            self.phys = phy(self.n_metal,
                            self.k_r,
                            self.n_wg,
                            self.n_clad,
                            self.mu,
                            self.scaling,
                            self.wavelength,
                            neff_array[i]) 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver='RHS', delta_n=self.delta_n_wg, idx_RHS=idx_RHS, val_RHS=val_RHS,  eigval_num=0, eliminate_excitation= self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list.append(FOM)#(FOM/self.FOM0)


        self.prop_const_list = np.array(FOM_list)
        plot_propagation_response(neff_array, neff0, neffT, np.array(FOM_list))

        from scipy.signal import argrelextrema

        maxima_index = argrelextrema(np.array(FOM_list), np.greater)

        maximima_beta = neff_array[maxima_index]

        print("-------------------------------------------------")
        print("The propagation constant for the maximum values for the FOM are:")
        print("-------------------------------------------------")
        print(maximima_beta)

    def prop_const_sweep_peak_heated(self, neffmin, neffmax, neff0, neffT, N, dVs, idx_RHS, val_RHS, idx_RHS_heat, val_RHS_heat):


        #self.phys = phy(self.n_metal,
        #                    self.k_r,
        #                    self.n_wg,
        #                    self.n_clad,
        #                    self.mu,
        #                    self.scaling,
        #                    self.wavelength,
        #                    self.delta) 

        #self.delta_n_wg = np.zeros_like(self.delta_n_wg)
        
        #dVs_hom = np.zeros_like(self.dVini)

        #self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities

        neff_array = np.linspace(neffmin, neffmax, N)
        FOM_list = []
        FOM_list_heated = []

        #print("---------------------------------------------------")
        #print("Calculating the FOM for different propagation constants for unheated device...")
        #print("---------------------------------------------------")
        #for i in tqdm(range(len(neff_array))):

        #    self.phys = phy(self.n_metal,
        #                    self.k_r,
        #                    self.n_wg,
        #                    self.n_clad,
        #                    self.mu,
        #                    self.scaling,
        #                    self.wavelength,
        #                    neff_array[i]) 

        #    _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver='RHS', delta_n=self.delta_n_wg, idx_RHS=idx_RHS, val_RHS=val_RHS,  eigval_num=0, eliminate_excitation= self.eliminate_excitation)
        #    if self.FOM_type == "linear":
        #        FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
        #
        #    elif self.FOM_type == "log":
        #        _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
        #    FOM_list.append(FOM)

        #plot_propagation_response(neff_array, neff0, neffT, np.array(FOM_list), self.scaling)

        #from scipy.signal import argrelextrema

        #maxima_index = argrelextrema(np.array(FOM_list), np.greater)

        #maximima_beta = neff_array[maxima_index]

        #print("-------------------------------------------------")
        #print("The propagation constant for the maximum values for the FOM are:")
        #print("-------------------------------------------------")
        #print(maximima_beta)

        self.physT = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.deltaT,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad
                        ) 

        _ = self.solve_heat(dVs, "RHS", idx_RHS_heat, val_RHS_heat)

        print("---------------------------------------------------")
        print("Calculating the FOM for different propagation constants for heated device...")
        print("---------------------------------------------------")
        for i in tqdm(range(len(neff_array))):

            self.phys = phy(self.n_metal,
                            self.k_r,
                            self.n_wg,
                            self.n_clad,
                            self.mu,
                            self.scaling,
                            self.wavelength,
                            neff_array[i]) 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.phys, self.filThr_EM, 'RHS', self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, 0, self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list_heated.append(FOM)  

        #plot_propagation_response_coupled(neff_array, neff0, neffT, np.array(FOM_list), np.array(FOM_list_heated),self.scaling)  
        self.prop_const_list = np.array(FOM_list_heated)
        plot_propagation_response(neff_array, neff0, neffT, np.array(FOM_list_heated))

        from scipy.signal import argrelextrema

        maxima_index = argrelextrema(np.array(FOM_list_heated), np.greater)

        maximima_beta = neff_array[maxima_index]

        print("-------------------------------------------------")
        print("The propagation constant for the maximum values for the FOM are:")
        print("-------------------------------------------------")
        print(maximima_beta)

        if self.logfile:

            create_logfile_sweep(self, idx_RHS, val_RHS, idx_RHS_heat, val_RHS_heat, neff_array, neff0, neffT, values=FOM_list, sweep_text="partial_heat", neff_heat=neff_array, values_heat=FOM_list_heated)

    def prop_const_sweep_peak_coupled(self, neffmin, neffmax, neff0, neffT, N, dVs, idx_RHS, val_RHS, idx_RHS_heat, val_RHS_heat):


        self.phys = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.delta,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        )

        self.delta_n_wg = np.zeros_like(self.delta_n_wg)
        
        #dVs_hom = np.zeros_like(self.dVini)

        #self.FOM0, _ = self.dis_0.objective_grad(dVs_hom, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys,self.filThr_EM, self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, self.eliminate_excitation) # we compute the FOM and sensitivities

        neff_array = np.linspace(neffmin, neffmax, N)
        FOM_list = []
        FOM_list_heated = []
        Efield_list = []
        Efield_list_heated = []

        print("---------------------------------------------------")
        print("Calculating the FOM for different propagation constants for unheated device...")
        print("---------------------------------------------------")
        for i in tqdm(range(len(neff_array))):

            self.phys = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        neff_array[i],
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver='RHS', delta_n=self.delta_n_wg, idx_RHS=idx_RHS, val_RHS=val_RHS,  eigval_num=0, eliminate_excitation= self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
        
            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list.append(FOM)
            Efield_list.append(np.max(self.dis_0.normE)**2)

        #plot_propagation_response(neff_array, neff0, neffT, np.array(FOM_list), self.scaling)

        from scipy.signal import argrelextrema

        maxima_index = argrelextrema(np.array(FOM_list), np.greater)

        maximima_beta = neff_array[maxima_index]

        print("-------------------------------------------------")
        print("The propagation constant for the maximum values for the FOM are:")
        print("-------------------------------------------------")
        print(maximima_beta)

        self.physT = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        self.deltaT,
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 


        _ = self.solve_heat(dVs, "RHS", idx_RHS_heat, val_RHS_heat)

        print("---------------------------------------------------")
        print("Calculating the FOM for different propagation constants for heated device...")
        print("---------------------------------------------------")
        for i in tqdm(range(len(neff_array))):

            self.physT = phy(self.n_metal,
                        self.k_r,
                        self.n_wg,
                        self.n_clad,
                        self.mu,
                        self.scaling,
                        self.wavelength,
                        neff_array[i],
                        self.dz,
                        k_wg= self.k_wg,
                        k_metal = self.k_metal,
                        k_clad = self.k_clad,
                        alpha = self.alpha
                        ) 
 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM,  self.physT, self.filThr_EM, 'RHS', self.delta_n_wg, self.dis_heat, idx_RHS, val_RHS, 0, self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list_heated.append(FOM)  
            Efield_list_heated.append(np.max(self.dis_0.normE)**2)

        plot_propagation_response_coupled(neff_array, neff0, neffT, np.array(FOM_list), np.array(FOM_list_heated))  
        #self.prop_const_list = np.array(FOM_list_heated)
        #plot_propagation_response(neff_array, neff0, neffT, np.array(FOM_list_heated), self.scaling)

        from scipy.signal import argrelextrema

        maxima_index = argrelextrema(np.array(FOM_list_heated), np.greater)

        maximima_beta = neff_array[maxima_index]

        print("-------------------------------------------------")
        print("The propagation constant for the maximum values for the FOM are:")
        print("-------------------------------------------------")
        print(maximima_beta)

        self.neff_array = neff_array
        self.FOM_list = FOM_list
        self.FOM_list_heated = FOM_list_heated
        self.E_list = Efield_list
        self.E_list_heated = Efield_list_heated


        if self.logfile:

            create_logfile_sweep(self, idx_RHS, val_RHS, idx_RHS_heat, val_RHS_heat, neff_array, neff0, neffT, values=FOM_list, sweep_text="total", neff_heat=neff_array, values_heat=FOM_list_heated)


    def frequency_sweep_peak(self, wl_cen, wl_bw, N, N_high, dVs, idx_RHS, val_RHS):

        """
        Function to perform the Topology Optimization based on a target FOM function.
        @ wl_cen: Central wavelength.
        @ wl_bw: Wavelength bandwidth. 
        @ N: Number of frequency points to be sampled in the range.
        """


        wl_array = np.linspace(wl_cen-0.5*wl_bw, wl_cen+0.5*wl_bw, N+1)
        wl_high_array = np.linspace(wl_cen-np.abs(wl_array[0]-wl_array[1]), wl_cen+np.abs(wl_array[0]-wl_array[1]), N_high+1)
        
        FOM_list = []

        print("---------------------------------------------------")
        print("Calculating the FOM for different wavelengths...")
        print("---------------------------------------------------")
        for i in tqdm(range(len(wl_array))):

            self.phys = phy(self.n_metal,
                            self.k_r,
                            self.n_wg,
                            self.n_clad,
                            self.mu,
                            self.scaling,
                            wl_array[i],
                            self.delta) 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver='RHS', delta_n=self.delta_n_wg , idx_RHS=idx_RHS, val_RHS=val_RHS,  eigval_num=0, eliminate_excitation= self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list.append(FOM/self.FOM0)

        print("---------------------------------------------------")
        print("Resolving the spectrum better close to central wavelength ...")
        print("---------------------------------------------------")

        for i in tqdm(range(len(wl_high_array))):

            self.phys = phy(self.n_metal,
                            self.k_r,
                            self.n_wg,
                            self.n_clad,
                            self.mu,
                            self.scaling,
                            wl_high_array[i],
                            self.delta) 

            _ = self.dis_0.FEM_sol(dVs, self.FOM_type, self.idxdr_EM, self.idx_wg_EM, self.phys, self.filThr_EM, solver='RHS', delta_n=self.delta_n_wg,  idx_RHS=idx_RHS, val_RHS=val_RHS,  eigval_num=0, eliminate_excitation= self.eliminate_excitation)
            if self.FOM_type == "linear":
                FOM, _, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)

            elif self.FOM_type == "log":
                _, FOM, __ = self.dis_0.compute_FOM(self.idx_wg_EM, self.eliminate_excitation, idx_RHS)
            FOM_list.append(FOM/self.FOM0)
        

        plot_frequency_response(wl_array, wl_high_array, wl_cen, np.array(FOM_list), self.scaling)

    def calculate_index_change(self, T, index_wg):

        dnSi_dT = 1.8E-4
            
        #T_wg = resize_to_el(np.reshape(T, (self.nElY_heat+1 ,self.nElX_heat+1)), self.nElX_heat+1 ,self.nElY_heat+1) [index_wg[0,:], index_wg[1,:]]
        #T_wg = np.reshape(T_wg, (self.nElYcore, self.nElXcore))
        #array_el = np.zeros(self.nElY_heat*self.nElX_heat)
        #for i in range(len(self.dis_heat.edofMat.astype(int))):
        #    nodes = self.dis_heat.edofMat.astype(int)[i, :]
        #    vals_nodes = T[nodes]
        #    sum = 0.0
        #    for val in vals_nodes:
        #        sum = sum  + val[0,0]
        #    array_el [i] = 0.25 * sum
        #    raise()

        #T_wg = np.zeros((self.nElY_EM, self.nElX_EM))
        #T_el = np.reshape(node_to_el(T, self.dis_heat.edofMat.astype(int), self.nElX_heat, self.nElY_heat), (self.nElY_heat, self.nElX_heat)) [index_wg[0,:], index_wg[1,:]]
        #T_wg [index_wg[0,:], index_wg[1,:]] = T_el
        
        #T_wg = np.reshape(T_wg, (self.nElYcore, self.nElXcore))
        T_wg = np.zeros((self.nElY_heat, self.nElX_heat, 4))
        T_el = np.zeros((self.nElY_heat * self.nElX_heat, 4))
        for i in range(len(self.dis_heat.edofMat.astype(int))):
            nodes = np.array(self.dis_heat.edofMat.astype(int))[i]
            T_el[i, :] = np.array(T).flatten()[nodes]

        T_el = np.reshape(T_el, (self.nElY_heat, self.nElX_heat, 4))
        T_wg [index_wg[0,:], index_wg[1,:], :] = T_el[index_wg[0,:], index_wg[1,:], :]

        delta_n = dnSi_dT * T_wg 

        #print(np.max(delta_n))
        #raise()

        return delta_n

    def plot_index_change(self):
        plot_n(self.delta_n_wg, self.dis_heat, self.nElYcore, self.nElXcore)

            
