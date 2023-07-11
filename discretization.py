from operator import index
from element_matrices import element_matrices
from material_interpolation import material_interpolation_metal
import scipy
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import use_solver
import numpy as np
from plot import plot_iteration
import time 
import matplotlib.pyplot as plt
from functions import effective_index
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit
from scikits.umfpack import spsolve, splu


def resize(A, nElx, nEly):
    import cv2
    A_T = np.reshape(A, (nEly, nElx)).T
    B_T = cv2.resize(A_T, dsize=(nEly+1, nElx+1), interpolation=cv2.INTER_LINEAR)
    return B_T.T



class dis:
    "Class that describes the discretized FEM model"
    def __init__(self, 
                 scaling,
                 nElx,
                 nEly,
                 debug=False):
        """
        @ scaling: scale of the physical problem; i.e. 1e-9 for nm.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ dVElmIdx: Indexes for the design variables.
        """

        self.scaling = scaling
        self.nElx = nElx
        self.nEly = nEly
        # -----------------------------------------------------------------------------------
        # INITIALIZE ELEMENT MATRICES
        # ----------------------------------------------------------------------------------- 
        self.E_mat, self.F_mat, self.BZT, self.BTZ, self.BZZ1, self.BZZ2= element_matrices(scaling)

        self.debug = debug

        use_solver(useUmfpack=True) # reset umfPack usage to default


    def index_set(self):
        
        """
        Sets indexes for:
        a) The system matrix: self.S
        b) The boundary conditions, self.n1BC, self.n2BC, self.n3BC, self.n4BC, self.nodes_per_line
        c) The right hand side (RHS)
        d) The full node matrix (where shared nodes are treated independently) used in the sensitivity calculation
        e) The sensitivity matrix which takes into account which nodes correspond to which elements in the indexing"
        """
        if self.debug:
            start = time.time()
        # -----------------------------------------------------------------------------------
        # A) SET INDEXES FOR THE SYSTEM MATRIX
        # ----------------------------------------------------------------------------------- 

        nEX = int(self.nElx) # Number of elements in X direction
        nEY = int(self.nEly) # Number of elements in Y direction

        self.nodesX = int(nEX + 1) # Number of nodes in X direction
        self.nodesY = int(nEY + 1) # Number of nodes in Y direction

        self.node_nrs = np.reshape(np.arange(0,self.nodesX * self.nodesY), (self.nodesY,self.nodesX)) # node numbering matrix
        self.node_nrs_flat = self.node_nrs.flatten() 

        self.elem_nrs = np.reshape(self.node_nrs[:-1,:-1], (nEY,nEX)) # element numbering matrix
        elem_nrs_flat = self.elem_nrs.flatten()

        self.N_edges =  nEX * (nEY+1) +  nEY * (nEX+1)

        self.nodes_per_elem = np.tile(elem_nrs_flat, (4,1)).T + np.ones((nEX*nEY,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1)) + self.N_edges

        self.num_nodes_Q8 = (2 * nEX +1) * (nEY + 1) + (nEX+1) * nEY
        num_nodes_Q8_X_1 = (2 * nEX +1)
        self.nodesX_new = num_nodes_Q8_X_1
        num_nodes_Q8_X_2 = (nEX+1)

        self.nodesY_new = num_nodes_Q8_X_2

        self.num_dof =  self.N_edges + self.num_nodes_Q8

        nodes_per_element_Q8_X2 =   np.reshape(np.repeat(elem_nrs_flat, 8),(nEX*nEY,8))  + np.ones((nEX*nEY,8))*np.tile(np.array([0, 0, 0, num_nodes_Q8_X_1+1, 0, 0, 0, num_nodes_Q8_X_1]), (nEX*nEY, 1)) + np.reshape(np.repeat(np.repeat(np.linspace(0,nEY-1,nEY) * (num_nodes_Q8_X_1),nEX),8), (nEX*nEY,8))

        nodes_per_element_Q8_X2 [:, 0] = 0.0
        nodes_per_element_Q8_X2 [:, 1] = 0.0
        nodes_per_element_Q8_X2 [:, 2] = 0.0
        nodes_per_element_Q8_X2 [:, 4] = 0.0
        nodes_per_element_Q8_X2 [:, 5] = 0.0
        nodes_per_element_Q8_X2 [:, 6] = 0.0
        elem_nrs_flat_rep_new = np.reshape(np.repeat(elem_nrs_flat, 8),(nEX*nEY,8)) 
        elem_nrs_flat_rep_new [:, 3] = 0.0
        elem_nrs_flat_rep_new [:, 7] = 0.0
        nodes_per_element_Q8_X1 = elem_nrs_flat_rep_new + np.reshape(np.tile(np.repeat(np.linspace(0,nEX-1,nEX),8),nEY), (nEX*nEY,8)) + np.ones((nEX*nEY,8))*np.tile(np.array([num_nodes_Q8_X_1+ num_nodes_Q8_X_2, num_nodes_Q8_X_1+ num_nodes_Q8_X_2+1 , num_nodes_Q8_X_1+ num_nodes_Q8_X_2+2, 0, 2, 1, 0, 0]), (nEX*nEY, 1))
        nodes_per_element_Q8_X1 [:, 3] = 0.0
        nodes_per_element_Q8_X1 [:, 7] = 0.0
        nodes_per_element_Q8 = nodes_per_element_Q8_X1 + nodes_per_element_Q8_X2
        A = np.reshape(np.repeat(np.repeat(np.linspace(0,nEY-1,nEY) * (num_nodes_Q8_X_1),nEX),8), (nEX*nEY,8))
        A [:, 3] = 0.0
        A [:, 7] = 0.0
        nodes_per_element_Q8 += A
        

        self.edofMat = nodes_per_element_Q8 + self.N_edges
        # to get all the combinations of nodes in elements we can use the following two lines:

        self.elem_nrs_old = np.reshape(self.node_nrs[:-1,:-1], (nEY,nEX)) # element numbering matrix
        elem_nrs_flat_old = self.elem_nrs.flatten()

        self.nodes_per_elem_old = np.tile(elem_nrs_flat_old, (4,1)).T + np.ones((nEX*nEY,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1))

        self.edofMat_old = np.tile(elem_nrs_flat_old, (4,1)).T + np.ones((nEY*nEX,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1)) # DOF matrix: nodes per element
        

        #print("NUMBER OF EDGES: ", self.N_edges)
        #print("NUMBER OF NODES: ", self.nodesX * self.nodesY)
        #print("NUMBER OF DOF: ", self.num_dof)

        #print("NODE NUMBERING: (MAX, MIN): ")
        #print(np.max(self.edofMat))
        #print(np.min(self.edofMat))

        self.iS = np.reshape(np.kron(self.edofMat,np.ones((8,1))), 64*self.nElx*self.nEly) # nodes in one direction
        self.iS_node = np.reshape(np.kron(self.edofMat,np.ones((4,1))), 32*self.nElx*self.nEly) # nodes in one direction
        self.jS = np.reshape(np.kron(self.edofMat,np.ones((1,8))), 64*self.nElx*self.nEly)  # nodes in the other direction
        self.jS_node = np.reshape(np.kron(self.edofMat,np.ones((1,4))), 32*self.nElx*self.nEly) # nodes in one direction
        
        #print(self.iS)
        #print(self.jS)
        #print(self.iS_node)
        #print(self.jS_node)

        #raise()
        

        edges_per_element = np.reshape(np.repeat(elem_nrs_flat, 4),(nEX*nEY,4))  + np.reshape(np.repeat(np.repeat(np.linspace(0,nEY-1,nEY) * nEX,nEX),4), (nEX*nEY,4)) + np.ones((nEX*nEY,4))*np.tile(np.array([nEX+(nEX+1), 0, nEX, nEX+1]), (nEX*nEY, 1)) 
        self.edges_per_element = edges_per_element

        self.edofMat_total = np.hstack([self.edofMat, self.edges_per_element]) # we get all the DOFs per element

        print("EDGE NUMBERING: (MAX, MIN): ")
        print(np.max(edges_per_element))
        print(np.min(edges_per_element))
        #print(edges_per_element)
        #raise()
        self.iS_edge = np.reshape(np.kron(edges_per_element,np.ones((4,1))), 16*self.nElx*self.nEly)    # edges in one direction
        self.iS_edge_node = np.reshape(np.kron(edges_per_element,np.ones((8,1))), 32*self.nElx*self.nEly)    # edges in one direction

        #print(self.iS_edge)
        self.jS_edge = np.reshape(np.kron(edges_per_element,np.ones((1,4))), 16*self.nElx*self.nEly)   # edges in the other direction
        self.jS_edge_node = np.reshape(np.kron(edges_per_element,np.ones((1,8))), 32*self.nElx*self.nEly)    # edges in one direction

        #print(self.iS_edge)
        #print(self.jS_edge)
        #raise()
        
        # -----------------------------------------------------------------------------------
        # B) SET INDEXES FOR THE BOUNDARY CONDITIONS
        # ----------------------------------------------------------------------------------- 

        end = self.num_nodes_Q8

        self.n1BC_Ez = np.arange(0,num_nodes_Q8_X_1) + self.N_edges
        self.n2BC_Ez = np.arange(0,end-num_nodes_Q8_X_1+1, num_nodes_Q8_X_1+num_nodes_Q8_X_2) + self.N_edges
        self.n2BC_Ez_2 = np.arange(num_nodes_Q8_X_1,end-num_nodes_Q8_X_1+1, num_nodes_Q8_X_1+num_nodes_Q8_X_2) + self.N_edges
        self.n3BC_Ez = np.arange(num_nodes_Q8_X_1-1,end, num_nodes_Q8_X_1+num_nodes_Q8_X_2) + self.N_edges
        self.n3BC_Ez_2 = np.arange(num_nodes_Q8_X_1+num_nodes_Q8_X_2-1,end, num_nodes_Q8_X_1+num_nodes_Q8_X_2) + self.N_edges
        self.n4BC_Ez = np.arange(end-num_nodes_Q8_X_1,end)  + self.N_edges


        self.n1BC_edge = np.arange(0, nEX) 
        self.n2BC_edge = np.arange(nEX, self.N_edges - nEX, 2*nEX+1)
        self.n3BC_edge = np.arange(2*nEX, self.N_edges, 2*nEX+1) 
        self.n4BC_edge = np.arange(self.N_edges-nEX, self.N_edges) 


        #self.nBC = np.concatenate([self.n1BC_Ez, self.n2BC_Ez,self.n3BC_Ez, self.n4BC_Ez]) # normal to boundary
        #self.nBC = np.concatenate([self.n1BC_edge, self.n2BC_edge,self.n3BC_edge, self.n4BC_edge, self.n1BC_Ez, self.n2BC_Ez, self.n2BC_Ez_2, self.n3BC_Ez, self.n3BC_Ez_2, self.n4BC_Ez])
        self.nBC = np.concatenate([self.n1BC_Ez, self.n2BC_Ez, self.n2BC_Ez_2, self.n3BC_Ez, self.n3BC_Ez_2, self.n4BC_Ez])


        # -----------------------------------------------------------------------------------
        # C) SET INDEXES FOR THE FULL NODE MATRIX
        # ----------------------------------------------------------------------------------- 
        
        # to match all elements with nodes (and vice versa) we flatten the DOF matrix
        self.edofMatfull = np.hstack([self.edofMat, self.edges_per_element])
        
        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in initialization: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        

    def system_RHS(self, phy, idx_RHS, val_RHS):
        """
        Sets the system's RHS.
        In this case, we count om having an incident plane wave from the RHS.
        @ phy:  Physical properties of the system.
        @ idx_RHS: Indexes of the RHS.
        @ val_RHS:
        """        
        F = np.zeros((self.num_dof,1), dtype="complex128") # system RHS

        for i in range(len(idx_RHS)):
        
            F[idx_RHS[i]] += val_RHS[i] * phy.kz # plane wave, this can be changed to modify results' scaling

        return F 
    
    def material_distribution(self, dVs, design_region_indexes):
        """
        Sets the material in the simulation domain.
        In this case, we set the air region, the substrate and the deisgn region.
        @ dVs: Values for the design variables.
        """ 

        dFP = np.zeros((self.nEly, self.nElx), dtype="complex128") # we initialize the domain to silicon oxide
        dFP [design_region_indexes[0,:], design_region_indexes[1,:]] = np.reshape(dVs, np.shape(dFP [design_region_indexes[0,:], design_region_indexes[1,:]]))
        
        return dFP
    
    def assemble_matrix(self, eps, phy):
        """
        Assembles the global system matrix.
        Since all our elements are linear rectangular elements we can use the same element matrices for all of the elements.
        @ eps: Value for the dielectric constant for all elements.
        @ phy: physics class objects that holds the physical parameters of the system..
        """ 
        if self.debug:
            start = time.time()

        eps_S_4 = np.repeat(eps.flatten(),16)
        eps_S_8 = np.repeat(eps.flatten(),64)

        self.ATT_S = np.tile(self.E_mat.flatten(),self.nElx*self.nEly) - phy.k**2*eps_S_4*np.tile(self.F_mat.flatten(),self.nElx*self.nEly)
        self.BTT_S = np.tile(self.F_mat.flatten(),self.nElx*self.nEly)
        self.BZT_S = np.tile(self.BZT.flatten(),self.nElx*self.nEly)
        self.BTZ_S = np.tile(self.BTZ.flatten(),self.nElx*self.nEly)
        self.BZZ_S = np.tile(self.BZZ1.flatten(),self.nElx*self.nEly) - phy.k**2*eps_S_8*np.tile(self.BZZ2.flatten(),self.nElx*self.nEly)
    
        #print(np.max(self.E_mat))
        #print(np.min(self.E_mat))
        #print(self.E_mat)
        #print(4.2*phy.k**2*self.F_mat)
        #print(self.E_mat - 4.2*phy.k**2*self.F_mat)
        #raise()
        #print(4.2*phy.k**2*np.min(self.F_mat))
        #print(np.max(self.ATT_S))
        #print(np.min(self.ATT_S))
        #print(np.max(self.BTT_S))
        #print(np.min(self.BTT_S))
        #print(np.max(self.BZT_S))
        #print(np.min(self.BZT_S))
        #print(np.max(self.BTZ_S))
        #print(np.min(self.BTZ_S))
        #print(np.max(self.BZZ1.flatten()))
        #print(np.min(self.BZZ1.flatten()))
        #print(self.BZZ1 - phy.k**2*4.2*self.BZZ2)
        #raise()
        #print(phy.k**2*4.2*np.min(self.BZZ2.flatten()))
        #print(np.max(self.BZZ_S))
        #print(np.min(self.BZZ_S))

        self.Phi = np.max(eps_S_4) * phy.k**2


        self.ATT_sparse = scipy.sparse.csr_matrix((self.ATT_S,(self.iS_edge.astype(int), self.jS_edge.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128')
        self.BTT_sparse = scipy.sparse.csr_matrix((self.BTT_S,(self.iS_edge.astype(int), self.jS_edge.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128') 
        self.BZT_sparse = scipy.sparse.csr_matrix((self.BZT_S,(self.iS_node.astype(int), self.jS_edge_node.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128')
        self.BTZ_sparse = scipy.sparse.csr_matrix((self.BTZ_S,(self.iS_edge_node.astype(int), self.jS_node.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128')
        self.BZZ_sparse = scipy.sparse.csr_matrix((self.BZZ_S,(self.iS.astype(int), self.jS.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128') 

        S = (self.ATT_sparse)
        S.eliminate_zeros()
        S.sum_duplicates()
        
       

        M = (self.BTT_sparse + self.BZZ_sparse + self.BZT_sparse + self.BTZ_sparse) 
        M.eliminate_zeros()
        M.sum_duplicates()

        
        self.A  = S + phy.kz**2*M

        boundary_values = np.ones_like(self.nBC)
        I = scipy.sparse.identity(n=self.num_dof, format ="csr", dtype='complex128')

        N = I - scipy.sparse.csr_matrix((boundary_values,(self.nBC.astype(int), self.nBC.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128')
        

        # apply dirichlet 0 boundary conditions with operations

        self.A = N.T @ self.A @ N + I - N 

        S = N.T @ S @ N + I - N 

        M = N.T @ M @ N 

        #self.A [:, self.nBC]  = 0.0
        #self.A [self.nBC, :]  = 0.0
        #self.A [self.nBC, self.nBC]  = 1.0

        # we sum all duplicates, which is equivalent of accumulating the value for each node
        # implement dirichlet boundary conditions as Jonathan sent
        #S.sum_duplicates()
        #S [:, self.nBC]  = 0
        #S [self.nBC, :]  = 0
        #S [self.nBC, self.nBC]  = 1
        #S.sum_duplicates()
        #S.eliminate_zeros()
        #.sum_duplicates()
        #M [self.nBC, :]  = 0
        #M [:, self.nBC]  = 0
        #M [self.nBC, self.nBC]  = 1
        #M.sum_duplicates()
        #M.eliminate_zeros()


        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in assembly: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        return S.tocsr(), M.tocsr()
    
    def solve_eigenproblem(self,phy):
        if self.debug:
            start = time.time()
        sigma =   phy.kz*np.conj(phy.kz)
        print('sigma: ', sigma)
        eigval , eigvec = scipy.sparse.linalg.eigs(A=self.S, k=5, M=self.M, sigma=-sigma)
        print(eigval)
        print(np.shape(eigvec))
        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in solving eigenproblem: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        return eigval, eigvec

    def solve_sparse_system(self, F, phy):
        """
        Solves a sparse system of equations using LU factorization.
        @ S: Global System matrix
        @ F: RHS array 
        """ 
        if self.debug:
            start = time.time()
        #lu = sla.splu(self.S-phy.k**2*self.M)
        #lu = sla.splu(self.A)
        lu = sla.splu(self.A)
        EzHz = lu.solve(F)
        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in solving linear system: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        return lu, EzHz

    def design_variable_derivative(self, dFPS):

        grad = np.gradient(dFPS)
        grad_x = grad[0]
        grad_y = grad[1]

        d_dFPS = np.sqrt(grad_x*np.conj(grad_x) + grad_y*np.conj(grad_y))

        d_dFPS_2 = grad_x*np.conj(grad_x) + grad_y*np.conj(grad_y)

        return np.reshape(d_dFPS, (self.nEly, self.nElx)), np.reshape(d_dFPS_2, (self.nEly, self.nElx))

    
    def FEM_sol(self, dVs, FOM_type, indexes_design_region, idx_wg, phy, filThr, solver, delta_n, dis_heat = None, idx_RHS = None, val_RHS=None, eigval_num=0, eliminate_excitation=False):
        """
        Gives the solution to the forward FEM problem; this is, the electric field solution.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        @ solver: either the RHS or eigenvalue solver.
        @ idx_RHS: element indexes where to apply the RHS
        @ vale_RHS: value to apply for the RHS in each element.
        @ eigval_num : Eiegenvalue to be selected out of all the calculated.
        """ 
        # -----------------------------------------------------------------------------------
        # FILTERING AND THRESHOLDING ON THE MATERIAL
        # ----------------------------------------------------------------------------------- 

        self.dFP = self.material_distribution(dVs, indexes_design_region)
        self.dFPS = filThr.density_filter(np.ones((self.nEly, self.nElx), dtype="complex128"), filThr.filSca, self.dFP, np.ones((self.nEly, self.nElx), dtype="complex128"))
        self.dFPST = filThr.threshold(self.dFPS)

        self.d_dFPS, self.d_dFPS_2 = self.design_variable_derivative(self.dFPS)

        # -----------------------------------------------------------------------------------
        #  CHAIN RULE FIELDS FOR GEOMETRIC CONSTRAINTS
        # -----------------------------------------------------------------------------------

        self.d_thresh = filThr.deriv_threshold(self.dFPS)
        
        self.d_thresh_filter = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), np.ones((self.nEly,self.nElx), dtype="complex128") , self.d_thresh)
        #self.d_thresh_filter = filThr.deriv_threshold(filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), np.ones((self.nEly,self.nElx), dtype="complex128") , np.ones((self.nEly,self.nElx), dtype="complex128")))

        #self.d_filter = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), np.ones((self.nEly,self.nElx), dtype="complex128"))
        self.d_filter = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), np.ones((self.nEly,self.nElx), dtype="complex128"), self.dFP)

        self.d_grad_filter = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), np.ones((self.nEly,self.nElx), dtype="complex128") , self.d_dFPS)

        #self.d_grad_filter, _ = self.design_variable_derivative(self.d_filter)
  
        # -----------------------------------------------------------------------------------
        # MATERIAL INTERPOLATION
        # ----------------------------------------------------------------------------------- 
        
        self.eps, self.depsdx, self.depsdT = material_interpolation_metal(phy.n_metal, phy.k_r, phy.n_wg, phy.n_clad, self.dFPST, indexes_design_region, idx_wg, delta_n, phy.alpha) 


        # -----------------------------------------------------------------------------------
        # SYSTEM RHS
        # -----------------------------------------------------------------------------------

        if solver == "RHS":
           
            F = self.system_RHS(phy,  idx_RHS, val_RHS)

        # -----------------------------------------------------------------------------------
        # ASSEMBLY OF GLOBAL SYSTEM MATRIX
        # -----------------------------------------------------------------------------------
        self.S, self.M = self.assemble_matrix(self.eps, phy)

        # -----------------------------------------------------------------------------------
        # SOLVE SYSTEM OF EQUATIONS
        # -----------------------------------------------------------------------------------
        if solver == 'RHS':

            self.lu, self.E = self.solve_sparse_system(F, phy)
            
            #grid = np.reshape(np.arange(0,self.nElx * self.nEly), (self.nEly,self.nElx))
            #element_number = grid[idx_wg[0,:], idx_wg[1,:]]

            #nodes_Ez =  self.edofMat[element_number,:] # get elements that are in the waveguide and from there get Ez dofs.
            #edges_Et =  self.edges_per_element[element_number,:] # get elements that are in the waveguide and from there get Ez dofs

            #self.E [:] = 0.0
            #self.E [nodes_Ez.astype(int)] = 1.0
            #self.E [edges_Et.astype(int)] = 1.0

            kz = phy.delta*phy.k

            #self.E[:self.N_edges] = self.E[:self.N_edges] /kz
            self.Et = self.E[:self.N_edges] / kz
            #self.Ez = self.E[self.N_edges:] / (-1j)
            indexes_nodes_4 = np.tile(np.arange(self.nodesX_new)[::2], self.nEly+1) + np.repeat((self.nodesX_new+self.nodesY_new)*np.arange(self.nEly+1),self.nElx+1) + self.N_edges
            
            #print(np.max(self.Et))
            #self.E[self.N_edges:] = self.E[self.N_edges:] / (-1j)
            self.Ez =  self.E[indexes_nodes_4] / (-1j)

            from element_matrices import evaluate_edge_at_el

            #self.Et = np.zeros(self.nElx*self.nEly)

            Et_edges = self.E[:self.N_edges] / kz
            self.Ex = np.zeros(self.nodesX*self.nodesY, dtype="complex128")
            self.Ey = np.zeros(self.nodesX*self.nodesY, dtype="complex128")
            i = 0
            for edges in self.edges_per_element:


                Et_e = (Et_edges[edges.astype(int)]).flatten()
                ets1 = Et_e [0]
                ets2 = Et_e [1]
                ets3 = Et_e [2]
                ets4 = Et_e [3]

                nodes = self.edofMat_old[i,:].astype(int)
            

                et_n1, et_n2, et_n3, et_n4 = evaluate_edge_at_el(self.scaling, ets1, ets2, ets3, ets4)
                self.Ex [nodes] = np.array([et_n1 [0], et_n2 [0], et_n3 [0], et_n4 [0]])
                self.Ey [nodes] = np.array([et_n1 [1], et_n2 [1], et_n3 [1], et_n4 [1]])

                i += 1
            #raise()
            

        if solver == 'eigenmode':

            self.eigvals, self.eigvecs = self.solve_eigenproblem(phy)

            print(np.shape(self.eigvecs))

            self.E = self.eigvecs[:, eigval_num]


            kz = np.sqrt(-self.eigvals[eigval_num])  

            self.E[:self.N_edges] = self.E[:self.N_edges] /kz
            self.Et = self.E[:self.N_edges]
            
            #self.Ez = self.E[self.N_edges:] / (-1j)

            indexes_nodes_4 = np.tile(np.arange(self.nodesX_new)[::2], self.nEly+1) + np.repeat((self.nodesX_new+self.nodesY_new)*np.arange(self.nEly+1),self.nElx+1) + self.N_edges

            self.Ez =  self.E[indexes_nodes_4] / (-1j)
            
            from element_matrices import evaluate_edge_at_el
            #self.Et = np.zeros(self.nElx*self.nEly)
            Et_edges = self.E[:self.N_edges]
            self.Ex = np.zeros(self.nodesX*self.nodesY)
            self.Ey = np.zeros(self.nodesX*self.nodesY)
            i = 0

            for edges in self.edges_per_element:


                Et_e = (Et_edges[edges.astype(int)]).flatten()
                ets1 = Et_e [0]
                ets2 = Et_e [1]
                ets3 = Et_e [2]
                ets4 = Et_e [3]

                nodes = self.edofMat_old[i,:].astype(int)
            

                et_n1, et_n2, et_n3, et_n4 = evaluate_edge_at_el(self.scaling, ets1, ets2, ets3, ets4)
                self.Ex [nodes] = np.array([et_n1 [0], et_n2 [0], et_n3 [0], et_n4 [0]])
                self.Ey [nodes] = np.array([et_n1 [1], et_n2 [1], et_n3 [1], et_n4 [1]])

                i += 1
            
            #omega = np.sqrt(self.eigvals[ eigval_num]) * 3e8
            print('The effective index calculated by the eigensolver is: ', kz/phy.k)

        
        self.normE = np.sqrt(self.Ex.flatten()*np.conj(self.Ex.flatten())+self.Ey.flatten()*np.conj(self.Ey.flatten())+self.Ez.flatten()*np.conj(self.Ez.flatten()))

        return self.E

    def get_lu_factorization_matrices(self):
        """
        Gives the LU factorization of a sparse matrix.
        Definitions from reference (scipy.sparse.linalg.SuperLU documentation), adjusted to case.
        """ 
        if self.debug:
            start = time.time()
        L = self.lu.L
        U = self.lu.U
        PR = scipy.sparse.csr_matrix((np.ones(self.num_dof, dtype="complex128"), (self.lu.perm_r, np.arange(self.num_dof))), dtype="complex128") # Row permutation matrix
        PC = scipy.sparse.csr_matrix((np.ones(self.num_dof, dtype="complex128"), (np.arange(self.num_dof), self.lu.perm_c)), dtype="complex128") # Column permutation matrix
        
        #PR = np.zeros((self.num_dof, self.num_dof), dtype="complex128")
        #PR[self.lu.perm_r, np.arange(self.num_dof)] = 1
        #PR = scipy.sparse.csc_matrix(PR)
        #PC = np.zeros((self.num_dof, self.num_dof), dtype="complex128")
        #PC[np.arange(self.num_dof), self.lu.perm_c] = 1
        #PC = scipy.sparse.csc_matrix(PC)

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in LU factorization: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        return L, U, PR, PC


    def compute_sensitivities(self, phy, eps, dAdx, E, AdjLambda,  AdjLambdaT, dis_heat):
        """
        Computes the sensitivities for all of the elements in the simulation domain.
        @ M: Mass matrix for each element
        @ k: wavevector of the problem (Frequency domain solver).
        @ dAdx: derivative of the design variables in the simulation domain. 
        @ Ez: electric field calculated from the forward problem.
        @ AdjLambda: Vector obtained by solving S.T * AdjLambda = AdjRHS
        """ 
        if self.debug:
            start = time.time()
            
        sens = np.zeros(self.nEly * self.nElx, dtype="complex128")
        sens_T = np.zeros(self.nEly * self.nElx, dtype="complex128")

        #@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
        #def calc_sens_coup_unheated(edofMatfull, edofMat, dxdx, dydy, dxdy, dyn, ndx, kz, k, delta_n, nn, dAdx_flat, E, AdjLambda, AdjLambdaT, LEM_heat, T, dFdx_e, dFdx, sens):
            # put edofmatfull astype int!

        #    for i in range(len(edofMatfull)):

        #        dAxx_e =  np.reshape(dxdx*dAdx_flat [i], (4,4))
        #        dAyy_e =  np.reshape(dydy*dAdx_flat [i], (4,4))
        #        dAzz_e =  np.reshape(kz**2 * nn*dAdx_flat [i], (4,4))
        #        dAxy_e =  np.reshape(dxdy*dAdx_flat [i], (4,4))
        #        dAyx_e =  np.reshape(dxdy.T*dAdx_flat [i], (4,4))
        #        dAyz_e =  np.reshape(kz * dyn *dAdx_flat [i], (4,4))
        #        dAzy_e =  np.reshape(kz * dyn.T *dAdx_flat [i], (4,4))
        #        dAzx_e =  np.reshape(kz * ndx *dAdx_flat [i], (4,4))
        #        dAxz_e =  np.reshape(kz * ndx.T *dAdx_flat [i], (4,4))

        #        dBx_e = np.reshape(nn *dAdx_flat [i], (4,4))

        #        dSdx_e = np.zeros((12,12), dtype="complex128")
        #        dSdx_e [:4,:4] =  dAxx_e - k**2 * dBx_e
        #        dSdx_e [4:8,4:8] =  dAyy_e - k**2 * dBx_e 
        #        dSdx_e [8:,8:] =  dAzz_e - k**2 * dBx_e
        #        dSdx_e [:4,4:8] = dAyx_e
        #        dSdx_e [4:8,:4] = dAxy_e
        #        dSdx_e [:4,8:] = dAzx_e
        #        dSdx_e [8:,:4] = dAxz_e
        #        dSdx_e [4:8,8:] = dAzy_e
        #        dSdx_e [8:,4:8] = dAyz_e

        #        AdjLambda_e = np.array([AdjLambda[n] for n in edofMatfull[i]])
        #        E_e = np.array([E[n] for n in edofMatfull[i]])

        #        sens [i] = 2*np.real((np.expand_dims(AdjLambda_e, 0) @ dSdx_e @ E_e) [0])

        #    return sens

        #@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
        #def calc_sens_coup_heated(edofMatfull, edofMat, dxdx, dydy, dxdy, dyn, ndx, kz, k, delta_n, nn, dAdx_flat, E, AdjLambda, AdjLambdaT, LEM_heat, T, dFdx_e, dFdx, sens):
            # put edofmatfull astype int!

        #    for i in range(len(edofMatfull)):

        #        dAxx_e =  np.reshape(dxdx*dAdx_flat [i], (4,4))
        #        dAyy_e =  np.reshape(dydy*dAdx_flat [i], (4,4))
        #        dAzz_e =  np.reshape(kz**2 * nn*dAdx_flat [i], (4,4))
        #        dAxy_e =  np.reshape(dxdy*dAdx_flat [i], (4,4))
        #        dAyx_e =  np.reshape(dxdy.T*dAdx_flat [i], (4,4))
        #        dAyz_e =  np.reshape(kz * dyn *dAdx_flat [i], (4,4))
        #        dAzy_e =  np.reshape(kz * dyn.T *dAdx_flat [i], (4,4))
        #        dAzx_e =  np.reshape(kz * ndx *dAdx_flat [i], (4,4))
        #        dAxz_e =  np.reshape(kz * ndx.T *dAdx_flat [i], (4,4))

        #        dBx_e = np.reshape(nn *dAdx_flat [i], (4,4))

        #        dSdx_e = np.zeros((12,12), dtype="complex128")
        #        dSdx_e [:4,:4] =  dAxx_e - k**2 * dBx_e
        #        dSdx_e [4:8,4:8] =  dAyy_e - k**2 * dBx_e 
        #        dSdx_e [8:,8:] =  dAzz_e - k**2 * dBx_e
        #        dSdx_e [:4,4:8] = dAyx_e
        #        dSdx_e [4:8,:4] = dAxy_e
        #        dSdx_e [:4,8:] = dAzx_e
        #        dSdx_e [8:,:4] = dAxz_e
        #        dSdx_e [4:8,8:] = dAzy_e
        #        dSdx_e [8:,4:8] = dAyz_e

        #        AdjLambda_e = np.array([AdjLambda[n] for n in edofMatfull[i]])
        #        E_e = np.array([E[n] for n in edofMatfull[i]]) 

        #        dSdx_T = dAdx_flat[i] * LEM_heat

        #        AdjLambda_T = np.array([AdjLambdaT[n] for n in edofMat[i]]).flatten()
        #        T_e = np.array([T[n] for n in edofMat[i]]).flatten()
        #        dFdx_e =  np.array([dFdx_e[i], dFdx_e[i], dFdx_e[i], dFdx_e[i]])

        #        sens_T[i] = np.real((np.expand_dims(AdjLambda_T, 0) @ ((dSdx_T @ T_e) - dFdx_e)) [0]) -  np.real((np.expand_dims(AdjLambda_T, 0) @ dFdx) [0])

        #        sens [i] = 2*np.real((np.expand_dims(AdjLambda_e, 0) @ dSdx_e @ E_e) [0]) 

        #    sens += sens_T

        #    return sens

        #if np.max(self.delta_n) == 0:
        #    sens = calc_sens_coup_unheated(self.edofMatfull.astype(int), dis_heat.edofMat.astype(int), self.dxdx, self.dydy, self.dxdy, self.dyn, self.ndx, phy.kz, phy.k, self.delta_n, self.nn, dAdx.flatten(), E.flatten(), AdjLambda, AdjLambdaT, dis_heat.LEM, dis_heat.T.flatten(), dis_heat.dFdx_e, dis_heat.dFdx, sens)
        #else: 
        #    sens = calc_sens_coup_heated(self.edofMatfull.astype(int), dis_heat.edofMat.astype(int), self.dxdx, self.dydy, self.dxdy, self.dyn, self.ndx, phy.kz, phy.k, self.delta_n, self.nn, dAdx.flatten(), E.flatten(), AdjLambda, AdjLambdaT, dis_heat.LEM, dis_heat.T.flatten(), dis_heat.dFdx_e, dis_heat.dFdx, sens)

        for i in range(len(self.edofMat_total)):


            dAtt_e =  -phy.k**2 * np.reshape(self.F_mat*dAdx.flatten() [i], (4,4))
            dBzz_e =  -phy.k**2 * np.reshape(self.BZZ2*dAdx.flatten() [i], (8,8))

            dSdx_e = np.zeros((12,12), dtype="complex128")
            dSdx_e [:8,:8] =  phy.kz**2 * dBzz_e
            dSdx_e [8:,8:] =  dAtt_e 


            AdjLambda_e = np.array([AdjLambda[n] for n in self.edofMat_total[i].astype(int)])
            E_e = np.array([E[n] for n in self.edofMat_total[i].astype(int)]).flatten()

            if np.max(self.delta_n) == 0:

                sens [i] = 2*np.real((AdjLambda_e[np.newaxis] @ dSdx_e @ E_e) [0])

            else:  

                dSdx_T = dis_heat.dAdx.flatten()[i] * dis_heat.LEM

                AdjLambda_T = np.array([AdjLambdaT[n] for n in dis_heat.edofMat[i].astype(int)])
                T_e = np.array([dis_heat.T[n] for n in dis_heat.edofMat[i].astype(int)]).flatten()
                dFdx_e =  np.array([dis_heat.dFdx_e[i], dis_heat.dFdx_e[i], dis_heat.dFdx_e[i],dis_heat.dFdx_e[i]])

                sens_T[i] = np.real((AdjLambda_T[np.newaxis] @ ((dSdx_T @ T_e) - dFdx_e)) [0]) -  np.real((AdjLambdaT[np.newaxis] @ dis_heat.dFdx) [0])

                sens [i] = 2*np.real((AdjLambda_e[np.newaxis] @ dSdx_e @ E_e) [0]) 

        sens += sens_T

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in sensitivity calculation: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
    
        return np.reshape(sens, (self.nEly, self.nElx))

    
    def compute_FOM(self, indexes_FOM, eliminate_excitation, idx_RHS,phy):
        """
        Computes the numerical value of the FOM.
        """ 
        if self.debug:
            start = time.time()

        P = scipy.sparse.csr_matrix((self.num_dof,self.num_dof), dtype='complex128') 

        P.setdiag(0. / self.num_dof)
        #P.todok()
        grid = np.reshape(np.arange(0,self.nElx * self.nEly), (self.nEly,self.nElx))
        element_number = grid[indexes_FOM[0,:], indexes_FOM[1,:]]
        
        nodes_Ez =  self.edofMat[element_number,:] # get elements that are in the waveguide and from there get Ez dofs.
        edges_Et =  self.edges_per_element[element_number,:] # get elements that are in the waveguide and from there get Ez dofs


        #nodes_E_field = range(0,len(self.node_nrs_flat))
        #P [:, :] = -10.0 / len(self.node_nrs_flat)


        P [nodes_Ez, nodes_Ez] = 1.0 / (self.num_dof)
        P [edges_Et, edges_Et] = 1.0 / (self.num_dof*phy.kz**2)

        #P = P.tocsr()

        #E_new = self.E
        #E_new [nodes_Ex] = 1E9
        #E_new [nodes_Ey] = 1E9
        #E_new [nodes_Ez] = 1E9

        #Ex = E_new[:self.nodesX*self.nodesY] 
        #Ey = E_new[self.nodesX*self.nodesY:2*self.nodesX*self.nodesY] 
        #Ez = E_new[2*self.nodesX*self.nodesY:]
        #plt.imshow(np.reshape(np.real(Ez), (self.nodesY, self.nodesX)))
        #plt.show()
        #raise()



        if eliminate_excitation:

            for i in range(len(idx_RHS)):
                P[idx_RHS[i], idx_RHS[i]] = 0.0

        P.eliminate_zeros()
        E_scaled = self.E

        E_scaled[self.N_edges:] = E_scaled[self.N_edges:] #/ (-1j) 
        E_scaled[:self.N_edges] = E_scaled[:self.N_edges] #/ phy.kz

        FOM1 = np.array(np.real(np.conj(E_scaled.T) @ P @ E_scaled)).flatten()[0] 
        FOM2 = np.log(FOM1)

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in computing FOM: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")

        return FOM1, FOM2, P

    def compute_T_Adj_Lambda(self, eps, phy, dis_heat):

        if self.debug:
            start = time.time()

        #eps_S = np.repeat(eps.flatten(),16)
        #deps_dT = np.repeat(self.depsdT.flatten(),16)
        #k_T = np.repeat(self.k_T,16)

        #A_T_factor =  (k_T**-2) * deps_dT  * (1.-(eps_S / k_T**2)) 
        #A_prime_T_factor = (phy.mu*phy.mu_0) * -(k_T**-4) * deps_dT / np.sqrt(phy.mu_0*phy.eps_0)
        #B_T_factor =  deps_dT
        #C_T_factor = (phy.delta / (3e8)) * -(k_T**-4) * deps_dT / np.sqrt(phy.mu_0*phy.eps_0)
        #C_prime_T_factor = (phy.delta / (3e8)) * -(k_T**-4) * deps_dT / phy.eps_0

        #A_S = A_T_factor*np.tile(self.A.flatten(),self.nElx*self.nEly)
        #A_prime_S = A_prime_T_factor*np.tile(self.A_prime.flatten(),self.nElx*self.nEly)
        #B_S = B_T_factor*np.tile(self.B.flatten(),self.nElx*self.nEly)
        #C_S = C_T_factor*np.tile(self.C.flatten(),self.nElx*self.nEly)
        #C_prime_S = C_prime_T_factor*np.tile(self.C_prime.flatten(),self.nElx*self.nEly)

        #self.A_T = scipy.sparse.csr_matrix((A_S,(self.iS1.astype(int), self.jS1.astype(int))), shape=(2*len(self.node_nrs_flat),2*len(self.node_nrs_flat)), dtype='complex128')
        #self.A_prime_T = scipy.sparse.csr_matrix((A_prime_S,(self.iS2.astype(int), self.jS2.astype(int))), shape=(2*len(self.node_nrs_flat),2*len(self.node_nrs_flat)), dtype='complex128')
        #self.B_T = scipy.sparse.csr_matrix((B_S,(self.iS1.astype(int), self.jS1.astype(int))), shape=(2*len(self.node_nrs_flat),2*len(self.node_nrs_flat)), dtype='complex128')
        #self.C_T = scipy.sparse.csr_matrix((C_S,(self.iS2.astype(int), self.jS1.astype(int))), shape=(2*len(self.node_nrs_flat),2*len(self.node_nrs_flat)), dtype='complex128')
        #self.C_prime_T = scipy.sparse.csr_matrix((C_prime_S,(self.iS2.astype(int), self.jS1.astype(int))), shape=(2*len(self.node_nrs_flat),2*len(self.node_nrs_flat)), dtype='complex128')

        #dSdT = (self.A_T + self.A_prime_T + self.C_T + self.C_prime_T).tolil() - phy.k**2 * (self.B_T).tolil()
        #dSdT.sum_duplicates()
        #dSdT [self.nBC, :]  = 0
        #dSdT [:, self.nBC]  = 0
        #dSdT [self.nBC, self.nBC]  = 1
        #dSdT.sum_duplicates()
        #dSdT = np.zeros((2*self.nodesX*self.nodesY, 2*self.nodesX*self.nodesY), dtype="complex128")

        AdjRHS = np.zeros((self.nodesX*self.nodesY), dtype="complex128")
        AdjRHS_nodes = np.zeros(4, dtype="complex128")
        
        
        for i in range(len(self.edofMat_total)):

            nodes_full = self.edofMat_total[i].astype(int)
            nodes = self.edofMat_old[i].astype(int)
            AdjLambda_e = np.array([self.AdjLambda[n] for n in nodes_full])
            E_e = np.array([self.E[n] for n in nodes_full])


            for j in range(4):
            #depsdT = self.depsdT.flatten()[i]
                depsdT = self.depsdT[:,:,j].flatten()[i]

                dAtt_e =  -phy.k**2 * np.reshape(self.F_mat*depsdT, (4,4))
                dBzz_e =  -phy.k**2 * np.reshape(self.BZZ2*depsdT, (8,8))

                dSdx_e = np.zeros((12,12), dtype="complex128")
                dSdx_e [:8,:8] =  phy.kz**2 * dBzz_e
                dSdx_e [8:,8:] =  dAtt_e 

                #dSdT [np.ix_(nodes,nodes)] += dSdx_e

                #print(np.shape(EzHz_e.T))
                #print(np.shape(dSdx_el.T))
                #print(np.shape(AdjLambda_e))
                AdjRHS_nodes [j] = 2*np.real(E_e.T@dSdx_e.T@AdjLambda_e).flatten()[0]
                
            
                #AdjRHS_e =2*np.real(EzHz_e.T@dSdx_el.T@AdjLambda_e).flatten()[0]
                #print(AdjRHS_e)
            AdjRHS[nodes] += 0.25*AdjRHS_nodes
            #print(AdjRHS_e)
            #AdjRHS[nodes] +=  0.25 * np.array([AdjRHS_e, AdjRHS_e, AdjRHS_e, AdjRHS_e])
        
        #AdjLambda_T = dis_heat.PR.T @ dis_heat.lu2.solve(dis_heat.lu1.solve(dis_heat.PC.T @ (-AdjRHS)))
        #AdjLambda_T  = dis_heat.PR.T @  scipy.sparse.linalg.spsolve(dis_heat.L.T, scipy.sparse.linalg.spsolve(dis_heat.U.T, dis_heat.PC.T @ (-AdjRHS)))
        AdjLambda_T  = dis_heat.PR.T @  spsolve(dis_heat.L.T, spsolve(dis_heat.U.T, dis_heat.PC.T @ (-AdjRHS)))

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in computing Adj_Lambda_T: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        return AdjLambda_T

    def filter_sens(self, filThr, sens):

        DdFSTDFS = filThr.deriv_threshold(self.dFPS)
        
        sensFOM = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"),sens,DdFSTDFS)

        return sensFOM


    def objective_grad(self, dVs, FOM_type, indexes_design_region, indexes_FOM,  phy, filThr, delta_n, dis_heat = None, idx_RHS= None, val_RHS= None, eliminate_excitation=False):
        """
        Evaluates the FOM via the forward FEM problem and calculates the design sensitivities.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        """ 
        
        start = time.time() # Measure the time to compute elapsed time when finished
        
        # -----------------------------------------------------------------------------------
        # SOLVE FORWARD PROBLEM
        # -----------------------------------------------------------------------------------
        self.delta_n = delta_n

        self.E = self.FEM_sol(dVs, FOM_type, indexes_design_region, indexes_FOM, phy, filThr, "RHS", delta_n, dis_heat, idx_RHS=idx_RHS, val_RHS=val_RHS, eliminate_excitation=eliminate_excitation)
        
        # -----------------------------------------------------------------------------------
        # COMPUTE THE FOM
        # -----------------------------------------------------------------------------------
        if FOM_type == "linear":
            FOM, _, P = self.compute_FOM(indexes_FOM, eliminate_excitation, idx_RHS,phy)

        elif FOM_type == "log":
            FOM1, FOM, P = self.compute_FOM(indexes_FOM, eliminate_excitation, idx_RHS,phy)

        print('FOM partial: ', FOM)

        
        # -----------------------------------------------------------------------------------
        #  ADJOINT OF RHS
        # -----------------------------------------------------------------------------------
        AdjRHS = np.zeros(self.num_dof, dtype="complex128")

        E_scaled = self.E

        E_scaled[self.N_edges:] = E_scaled[self.N_edges:] #/ (1) 
        E_scaled[:self.N_edges] = E_scaled[:self.N_edges] #/ phy.kz*2


        if FOM_type == "linear":
            AdjRHS = (2 * np.real(E_scaled) - 1j* 2*np.imag(E_scaled)).flatten()


        elif FOM_type == "log": 
            precoeff = 1.0 / FOM1 
            AdjRHS = (precoeff * ((2 * np.real(E_scaled)) - 1j* 2*np.imag(E_scaled))).flatten() 

        #print(np.max(AdjRHS[self.N_edges:]))
        #print(np.min(AdjRHS[self.N_edges:]))
        #print(np.max(AdjRHS[:self.N_edges]))
        #print(np.max(AdjRHS[:self.N_edges]))
        #raise()
    

        AdjRHS =  P @ AdjRHS 
        # -----------------------------------------------------------------------------------
        #  SOLVE THE ADJOINT SYSTEM: S.T * AdjLambda = AdjRHS
        # -----------------------------------------------------------------------------------
        L, U, PR, PC = self.get_lu_factorization_matrices()
        if self.debug:
            start = time.time()
        #scipy.sparse.linalg.use_solver(useUmfpack=False)
        #lu1 = sla.splu(U.T)
        #lu2 = sla.splu(L.T)
        #self.AdjLambda = PR.T @ lu2.solve(lu1.solve(PC.T @ (-0.5*AdjRHS)))
        
        #self.AdjLambda  = PR.T @  scipy.sparse.linalg.spsolve(L.T, scipy.sparse.linalg.spsolve(U.T, PC.T @ (-0.5*AdjRHS)))
        self.AdjLambda  = PR.T @  spsolve(L.T, spsolve(U.T, PC.T @ (-0.5*AdjRHS)))
        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in computing Adj_Lambda: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        # -----------------------------------------------------------------------------------
        #  SOLVE THE ADJOINT SYSTEM FOR COUPLING: S.T * AdjLambda = AdjRHS
        # -----------------------------------------------------------------------------------
        if np.max(self.delta_n) == 0:
            self.AdjLambda_T = None
        else:
            self.AdjLambda_T = self.compute_T_Adj_Lambda(self.eps, phy, dis_heat)

        # -----------------------------------------------------------------------------------
        #  COMPUTE SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        self.sens = self.compute_sensitivities(phy, self.eps, self.depsdx, self.E, self.AdjLambda, self.AdjLambda_T, dis_heat)
        # -----------------------------------------------------------------------------------
        #  FILTER  SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        sensFOM = self.filter_sens(filThr, self.sens)

        # -----------------------------------------------------------------------------------
        #  SENSITIVITIES FOR DESIGN REGION
        # -----------------------------------------------------------------------------------
        sensFOM = sensFOM [indexes_design_region[0,:], indexes_design_region[1,:]] 

        # -----------------------------------------------------------------------------------
        #  FOM FOR MINIMIZATION
        # -----------------------------------------------------------------------------------

        FOM = FOM
        sensFOM =  sensFOM

        # Plotting and printing per optimization iteration
        end = time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        print("Elapsed time in iteration: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        print("----------------------------------------------")
        #plot_iteration(self)
        
        return FOM, sensFOM





