from element_matrices import element_matrices_heat, boundary_element_matrix, volume_RHS_matrix
from material_interpolation import material_interpolation_heat
import scipy
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import use_solver
import numpy as np
from plot import plot_iteration, plot_mi
import time
import matplotlib.pyplot as plt
from numba import jit
from scikits.umfpack import spsolve, splu


class dis_heat:
    "Class that describes the discretized FEM model"
    def __init__(self, 
                 scaling,
                 nElx,
                 nEly,
                 debug):
        """
        @ scaling: scale of the physical problem; i.e. 1e-9 for nm.
        @ nElX: Number of elements in the X axis.
        @ nElY: Number of elements in the Y axis.
        @ tElmIdx: Target element's index for FOM calculation.
        @ dVElmIdx: Indexes for the design variables.
        """

        self.scaling = scaling
        self.nElx = nElx
        self.nEly = nEly
        # -----------------------------------------------------------------------------------
        # INITIALIZE ELEMENT MATRICES
        # ----------------------------------------------------------------------------------- 
        LEM, MEM = element_matrices_heat(scaling) 
        self.LEM = LEM
        self.MEM = MEM

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
        
        # -----------------------------------------------------------------------------------
        # A) SET INDEXES FOR THE SYSTEM MATRIX
        # ----------------------------------------------------------------------------------- 

        if self.debug:
            start = time.time()

        nEX = self.nElx # Number of elements in X direction
        nEY = self.nEly # Number of elements in Y direction

        self.nodesX = nEX + 1 # Number of nodes in X direction
        self.nodesY = nEY + 1 # Number of nodes in Y direction

        self.node_nrs = np.reshape(np.arange(0,self.nodesX * self.nodesY), (self.nodesY,self.nodesX)) # node numbering matrix
        self.node_nrs_flat = self.node_nrs.flatten() 

        self.elem_nrs = np.reshape(self.node_nrs[:-1,:-1], (nEY,nEX)) # element numbering matrix
        self.elem_nrs_flat = self.elem_nrs.flatten()


        self.edofMat = np.tile(self.elem_nrs_flat, (4,1)).T + np.ones((nEY*nEX,4))*np.tile(np.array([0, 1, nEX+1, nEX+2]), (nEX*nEY, 1)) # DOF matrix: nodes per element

        # to get all the combinations of nodes in elements we can use the following two lines:

        self.iS = np.reshape(np.kron(self.edofMat,np.ones((4,1))), 16*self.nElx*self.nEly) # nodes in one direction
        self.jS = np.reshape(np.kron(self.edofMat,np.ones((1,4))), 16*self.nElx*self.nEly) # nodes in the other direction
        
        # -----------------------------------------------------------------------------------
        # B) SET INDEXES FOR THE BOUNDARY CONDITIONS
        # ----------------------------------------------------------------------------------- 

        end = self.nodesX * self.nodesY # last node number

        self.n1BC = np.arange(0,self.nodesX) # nodes top
        self.n2BC = np.arange(0,end-self.nodesX+1, self.nodesX) #left
        self.n3BC = np.arange(self.nodesX-1,end, self.nodesX) #right
        self.n4BC = np.arange(end-self.nodesX,end) #bottom

        self.nBC = np.concatenate([self.n1BC, self.n2BC, self.n3BC, self.n4BC])
        self.nBC_const_heat = np.concatenate([self.n1BC])

        # For the implementation of the BC into the global system matrix we need to know which nodes each boundary line has:

        self.nodes_line1 = np.tile(self.n1BC[:-1], (2,1)).T + np.ones((len(self.n1BC)-1,2))*np.tile(np.array([0, 1]), (len(self.n1BC)-1, 1))
        self.nodes_line2 = np.tile(self.n2BC[:-1], (2,1)).T + np.ones((len(self.n2BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n2BC)-1, 1))
        self.nodes_line3 = np.tile(self.n3BC[:-1], (2,1)).T + np.ones((len(self.n3BC)-1,2))*np.tile(np.array([0, nEX+1]), (len(self.n3BC)-1, 1))
        self.nodes_line4 = np.tile(self.n4BC[:-1], (2,1)).T + np.ones((len(self.n4BC)-1,2))*np.tile(np.array([0,1]), (len(self.n4BC)-1, 1))

        self.lines = np.arange(0, 2 * (self.nElx + self.nEly))
        self.nodes_per_line = np.concatenate([self.nodes_line1,self.nodes_line2,self.nodes_line3,self.nodes_line4]) 

         # to get all the combinations of nodes in lines we can use the following two lines:

        self.ibS = np.reshape(np.kron(self.nodes_per_line,np.ones((2,1))), 8*(self.nElx+self.nEly))
        self.jbS = np.reshape(np.kron(self.nodes_per_line,np.ones((1,2))), 8*(self.nElx+self.nEly)) 

        # -----------------------------------------------------------------------------------
        # C) SET INDEXES FOR THE RHS
        # ----------------------------------------------------------------------------------- 

        RHSB = self.n4BC # we select the boundary corresponding to the RHS
        self.nRHS1 = RHSB[1:]  #shared nodes
        self.nRHS2 = RHSB[:-1] #shared nodes

        self.nRHS = np.array([self.nRHS1, self.nRHS2])

        # -----------------------------------------------------------------------------------
        # D) SET INDEXES FOR THE FULL NODE MATRIX
        # ----------------------------------------------------------------------------------- 

        # to match all elements with nodes (and vice versa) we flatten the DOF matrix

        self.idxDSdx = self.edofMat.astype(int).flatten()

        # to get all the combinations of nodes in elements we can use the following two lines:

        ima0 = np.tile([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3],(self.nElx*self.nEly)) 
        jma0 = np.tile([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],(self.nElx*self.nEly))

        addTMP = np.reshape(np.tile(4*np.arange(0,self.nElx*self.nEly),16),(16, self.nElx*self.nEly)).T.flatten()

        # by adding addTMP to ima0 and jma0 we can be sure that the indexes for each node are different so we get all node combinations
        # independently. This means that if there are two elements that share a node, this will not be summed in a matrix position, but
        # taken independently.

        self.iElFull = ima0 + addTMP
        self.jElFull = jma0 + addTMP

        # -----------------------------------------------------------------------------------
        # E) SET INDEXES FOR THE SENSITIVITY MATRIX
        # ----------------------------------------------------------------------------------- 

        # now we want to index all the nodes in the elements  

        self.iElSens = np.arange(0,4*self.nElx*self.nEly)
        self.jElSens = np.reshape(np.tile(np.arange(0,self.nElx*self.nEly),4),(4, self.nElx*self.nEly)).T.flatten()

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in initialization: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")

        

    def system_RHS(self, idx_RHS, val_RHS, dz, T_bound, phy):
        """
        Sets the system's RHS.
        In this case, we count om having an incident plane wave from the RHS.
        @ phy:  Physical properties of the system.
        @ idx_RHS: Indexes of the RHS.
        @ val_RHS:
        
        """    

        if self.debug:
            start = time.time()


        F =  np.zeros((self.nodesY*self.nodesX,1), dtype="complex128") # system RHS
        self.dFdx =  np.zeros((self.nodesY*self.nodesX,1), dtype="complex128") # derivative system RHS
        self.dFdx_e =  np.zeros((self.nElx*self.nEly), dtype="complex128") # derivative system RHS
        self.dFde=  np.zeros((self.nElx*self.nEly), dtype="complex128") # derivative system RHS
        v_e =    self.scaling**2 * dz 
        tot_mass = np.sum(self.dFPST.flatten()) * v_e
        volume =  len(self.dFPST.flatten()) * self.scaling**2 * dz   #(len(idx_RHS[0,:]) * self.scaling**2 * dz)
        Q_volume = val_RHS[0,0] / tot_mass # volume


        p2 = 3.0

        test = np.zeros((self.nEly , self.nElx))

        val_RHS_node = np.array(volume_RHS_matrix(self.scaling, Q_volume)) #volume_RHS_matrix(self.scaling, Q_volume)[0]

        #print(val_RHS_node)
        val_RHS_node_new = np.array(volume_RHS_matrix(self.scaling, v_e/tot_mass))
        #print(val_RHS_node_new)
        #raise()

        @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
        def assemble_F(F, dFdx, edofmat, dFPST, val_RHS_node, dAdx, L, val_RHS, p2):
            for i in range(len(edofmat)):
                nodes = np.array([int(node) for node in edofmat[i]])
                F[nodes] +=  dFPST.flatten()[i]**p2 * val_RHS_node
                dFdx[nodes] +=  val_RHS_node[0]  * (-dFPST.flatten()[i]**p2/np.sum(dFPST.flatten())) #val_RHS_node * (-val_RHS_node_new*dFPST[i] + np.ones(4))
            return F, dFdx


        #for i in range(len(self.edofMat)):

            #element =  self.elem_nrs [idx_RHS[0,i], idx_RHS[1,i]]
            #idx_element = np.where(self.elem_nrs_flat==element)
            #nodes = self.edofMat[idx_element].astype(int)
            #nodes = self.edofMat[i].astype(int)
            #test  [idx_RHS[0,i], idx_RHS[1,i]]  =  self.dFPST[idx_RHS[0,i], idx_RHS[1,i]]
            #F[nodes] +=  self.dFPST[idx_RHS[0,i], idx_RHS[1,i]] * val_RHS_node
            #F[nodes] +=  self.dFPST.flatten()[i] * val_RHS_node
            #self.dFPS_F [idx_RHS[0,i], idx_RHS[1,i]] = self.dFPS   [idx_RHS[0,i], idx_RHS[1,i]]
            #F[nodes] +=   val_RHS_node
            #self.dFdx[nodes] += val_RHS_node * (1-self.dFPST.flatten()[i]/tot_mass) 
            
        F, self.dFdx, = assemble_F(F, self.dFdx, self.edofMat, self.dFPST, val_RHS_node, self.dAdx, self.LEM, val_RHS, p2)
        #self.dFde = val_RHS_node[0]  * (-self.dFPST.flatten()**p2/np.sum(self.dFPST.flatten()))

        #self.dFdx_e[:] = np.real(val_RHS_node [0] * (-self.dFPST.flatten()/np.sum(self.dFPST.flatten()) + np.ones_like(self.dFPST.flatten())))

        #self.dFdx_e[:] = val_RHS_node [0]

        self.dFdx_e[:] = val_RHS_node [0] *p2* (self.dFPST.flatten()**(p2-1))
        #self.dFdx[:] =  val_RHS_node [0] * (1-0*0.75/np.sum(self.dFPST.flatten()))
        
        #for i in range(len(self.nBC_const_heat)):
        #    F -=  self.S[:, self.nBC_const_heat[i]] * T_bound #prescribed BC 
        
        #self.dFdx -=

            
        F [self.nBC_const_heat] = T_bound
        #self.S [self.nBC_const_heat, :]  = 0
        #self.S [:, self.nBC_const_heat]  = 0
        #self.S [self.nBC_const_heat,self.nBC_const_heat]  = 1

        I = scipy.sparse.identity(n=(len(self.node_nrs_flat)), format ="csr", dtype='complex128')

        values = np.ones_like(self.nBC_const_heat)

        N = I - scipy.sparse.csr_matrix((values,(self.nBC_const_heat.astype(int), self.nBC_const_heat.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        

        # apply dirichlet 0 boundary conditions with operations

        self.S = N.T @ self.S @ N + I - N 


        #self.dFdx[self.nBC_const_heat] =  0

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in RHS: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")

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
    
    def assemble_matrix(self, L, K, k, d_z):
        """
        Assembles the global system matrix.
        Since all our elements are linear rectangular elements we can use the same element matrices for all of the elements.
        @ L: Laplace matrix for each element
        @ M: Mass matrix for each element
        @ K: Boundary matrix for each line in the boundary
        @ k: wavevector of the problem (Frequency domain solver).
        @ eps: design variables in the simulation domain. 
        """ 

        if self.debug:
            start = time.time()


        L_S = np.tile(L.flatten(),self.nElx*self.nEly) # create 1D system Laplace array
        k_S = np.repeat(k, 16)
        self.vS = k_S * L_S
        # we can take all these values and assign them to their respective nodes
        S = scipy.sparse.csr_matrix((self.vS,(self.iS.astype(int), self.jS.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype="complex128")
        # we sum all duplicates, which is equivalent of accumulating the value for each node
        S.sum_duplicates()
        S.eliminate_zeros()
        #S = S.tolil()
        
        # we follow a similar process for the boundaries; however, now we go through the nodes on each line in the boundaries
        self.bS = np.tile(K.flatten(),2*(self.nElx+self.nEly))
        K = scipy.sparse.csr_matrix((self.bS,(self.ibS.astype(int), self.jbS.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        K.sum_duplicates()
        boundaries =  self.n4BC #np.concatenate([self.n1BC, self.n2BC,self.n3BC])
        #K[boundaries,:] = 0.0
        #K.tocsc()
        #K[:,boundaries] = 0.0

         #we sum the contribution from the boundary to global the system matrix
        S = S + K

        boundary_values = np.ones_like(boundaries)
        I = scipy.sparse.identity(n=(len(self.node_nrs_flat)), format ="csr", dtype='complex128')

        N = I - scipy.sparse.csr_matrix((boundary_values,(boundaries.astype(int), boundaries.astype(int))), shape=(len(self.node_nrs_flat),len(self.node_nrs_flat)), dtype='complex128')
        

        # apply dirichlet 0 boundary conditions with operations

        self.S = N.T @ S @ N + I - N 

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in assembly: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")

        return S
    
    def solve_sparse_system(self,F):
        """
        Solves a sparse system of equations using LU factorization.
        @ S: Global System matrix
        @ F: RHS array 
        """ 

        if self.debug:
            start = time.time()


        lu = sla.splu(self.S)
        Ez = lu.solve(F)

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in solving system: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")

        return lu, Ez

    
    def FEM_sol_heat(self,dVs, FOM_type, indexes_design_region, idx_wg, phy, filThr, solver, idx_RHS = None, val_RHS=None, eigval_num=0, eliminate_excitation=False):
        """
        Gives the solution to the forward FEM problem; this is, the electric field solution.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        """ 
        # -----------------------------------------------------------------------------------
        # FILTERING AND THRESHOLDING ON THE MATERIAL
        # ----------------------------------------------------------------------------------- 
        self.dFP = self.material_distribution(dVs,indexes_design_region)
        self.dFPS = filThr.density_filter(np.ones((self.nEly, self.nElx), dtype="complex128"), filThr.filSca,  self.dFP, np.ones((self.nEly, self.nElx), dtype="complex128"))
        self.dFPST = filThr.threshold(self.dFPS)
        # -----------------------------------------------------------------------------------
        # MATERIAL INTERPOLATION
        # ----------------------------------------------------------------------------------- 

        self.idx_wg = idx_wg

        self.A, self.dAdx = material_interpolation_heat(phy.k_metal, phy.k_wg, phy.k_clad, self.dFPST, idx_wg)

        # -----------------------------------------------------------------------------------
        # ASSEMBLY OF GLOBAL SYSTEM MATRIX
        # -----------------------------------------------------------------------------------
        h = 10
        K_s = boundary_element_matrix(self.scaling, h, phy.dz)
        self.S = self.assemble_matrix(self.LEM, K_s, self.A, phy.dz)

        # -----------------------------------------------------------------------------------
        # SYSTEM RHS
        # -----------------------------------------------------------------------------------

        F = self.system_RHS(idx_RHS, val_RHS, phy.dz, 0, phy)

        # -----------------------------------------------------------------------------------
        # SOLVE SYSTEM OF EQUATIONS
        # -----------------------------------------------------------------------------------
        self.lu, self.T = self.solve_sparse_system(F)

        return self.T

    def get_lu_factorization_matrices(self):
        """
        Gives the LU factorization of a sparse matrix.
        Definitions from reference (scipy.sparse.linalg.SuperLU documentation), adjusted to case.
        """ 
        L = self.lu.L
        U = self.lu.U

        PR = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY, dtype="complex128"), (self.lu.perm_r, np.arange(self.nodesX*self.nodesY))), dtype="complex128") # Row permutation matrix
        PC = scipy.sparse.csc_matrix((np.ones(self.nodesX*self.nodesY, dtype="complex128"), (np.arange(self.nodesX*self.nodesY), self.lu.perm_c)), dtype="complex128") # Column permutatio

        #PR = np.zeros((self.nodesX*self.nodesY, self.nodesX*self.nodesY), dtype="complex128")
        #PR[self.lu.perm_r, np.arange(self.nodesX*self.nodesY)] = 1
        #PR = scipy.sparse.csc_matrix(PR)
        #PC = np.zeros((self.nodesX*self.nodesY, self.nodesX*self.nodesY), dtype="complex128")
        #PC[np.arange(self.nodesX*self.nodesY), self.lu.perm_c] = 1
        #PC = scipy.sparse.csc_matrix(PC)

        return L, U, PR, PC
    def compute_sensitivities(self, L, dAdx, T, AdjLambda):
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

        sens = np.zeros(self.nEly * self.nElx)
       
        F_es = np.zeros(self.nEly * self.nElx)
        for i in range(len(self.edofMat)):

            dSdx_e = dAdx.flatten()[i] * L

            #for j in range(len(self.edofMat[i].astype(int))):
            #    n = self.edofMat[i].astype(int) [j]
            #    if n in self.nBC_const_heat:
            #        dSdx_e [j, :]  = 0
            #        dSdx_e [:, j]  = 0
            #        dSdx_e [j,j]  = 1

            AdjLambda_e = np.array([AdjLambda[n] for n in self.edofMat[i].astype(int)])
            T_e = np.array([T[n] for n in self.edofMat[i].astype(int)]).flatten()
            #dFdx_e =  np.array([self.dFdx[n] for n in self.edofMat[i].astype(int)]).flatten()
            #print(self.dFdx_e[i])
            dFdx_e =  np.array([self.dFdx_e[i], self.dFdx_e[i], self.dFdx_e[i],self.dFdx_e[i]])
            #for j in range(len(self.edofMat[i].astype(int))):
            #    n = (self.edofMat[i])[j]
            #    if n in self.nBC_const_heat:
            #        dFdx_e [j] = 0.0
            #F_es[i] = np.sum(dFdx_e)
    
            #sens [i] = np.real((AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e))) [0]) + np.real((AdjLambda[np.newaxis] @ self.dFdx) [0])
            #print(np.sum(self.dFPST))
            #print(np.real((AdjLambda[np.newaxis] @ self.dFdx) [0]))
            #print(np.shape(AdjLambda_e[np.newaxis]))
            #sens [i] = np.real((AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e) - dFdx_e)) [0]) -  np.real((AdjLambda[np.newaxis] @ self.dFdx) [0])
            #sens [i] = np.real((AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e))) [0]) -  np.real((AdjLambda[np.newaxis] @ self.dFdx) [0])
            sens [i] = np.real((AdjLambda_e[np.newaxis] @ ((dSdx_e @ T_e) - dFdx_e)) [0]) -  np.real((AdjLambda[np.newaxis] @ self.dFdx) [0])


        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in computing sensitivities: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")
        
        return np.reshape(sens, (self.nEly, self.nElx))


    
    def compute_FOM(self, indexes_FOM, phy):
        """
        Computes the numerical value of the FOM.
        """ 

        if self.debug:
            start = time.time()

        self.P =  0.0 * scipy.sparse.dok_matrix(np.atleast_2d(np.zeros(len(self.node_nrs_flat))))
        nodes =  self.node_nrs[indexes_FOM[0,:], indexes_FOM[1,:]]


        self.P[:,nodes] = 1. / (len(nodes))
        self.P = self.P.tocsr()
        self.delta_T_pi = (1/2E-4) * 0.5*(phy.wavelength * phy.scale)/phy.dz 

        #print("Temperature change for pi-shift: ", self.delta_T_pi)
        #self.sign = False
        #if np.array(np.real(self.T).T @ self.P.todense().T).flatten()[0] - 0 - delta_T_pi > 0:
        #    self.sign = True

        self.T_wg =  np.array(np.real(self.T).T @ self.P.todense().T).flatten()[0]

        FOM =  (np.array(np.real(self.T).T @ self.P.todense().T).flatten()[0] - self.delta_T_pi)**2 # CHANGE RHS BASED ON THIS!!!!

        if self.debug:
            end = time.time()
            elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
            print("----------------------------------------------")
            print("Elapsed time in computing FOM: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
            print("----------------------------------------------")


        return FOM



    def objective_grad(self, dVs, FOM_type, indexes_design_region, indexes_FOM,  phy, filThr, idx_RHS, val_RHS, eliminate_excitation=False):
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
        solver = "RHS"
        self.T = self.FEM_sol_heat(dVs, FOM_type, indexes_design_region, indexes_FOM, phy, filThr, solver, idx_RHS = idx_RHS, val_RHS=val_RHS)
        # -----------------------------------------------------------------------------------
        # COMPUTE THE FOM
        # -----------------------------------------------------------------------------------
        self.indexes_FOM = indexes_FOM
        FOM = self.compute_FOM(indexes_FOM, phy)

        #print("FOM partial heat: ", FOM)
        # -----------------------------------------------------------------------------------
        #  ADJOINT OF RHS
        # -----------------------------------------------------------------------------------
        AdjRHS = np.zeros(self.nodesX*self.nodesY, dtype="complex128")
        #nodes =  self.node_nrs[indexes_FOM[0,:], indexes_FOM[1,:]]
        #AdjRHS = (2 * np.real(self.T))
        AdjRHS = np.array(2* np.array(np.real(self.T).T @ self.P.todense().T).flatten()[0] * self.P.todense().T -   2*self.P.todense().T*self.delta_T_pi).flatten()

        #print(np.shape(self.P))
        #print(np.shape(AdjRHS))
        #AdjRHS =  self.P @ AdjRHS 
        #nodes =  self.node_nrs[indexes_design_region[0,:], indexes_design_region[1,:]]
        #AdjRHS =  np.zeros(len(self.node_nrs_flat))
        #AdjRHS [nodes] = 1.0   / len(nodes)
        # -----------------------------------------------------------------------------------
        #  SOLVE THE ADJOINT SYSTEM: S.T * AdjLambda = AdjRHS
        # -----------------------------------------------------------------------------------

        self.L, self.U, self.PR, self.PC = self.get_lu_factorization_matrices()
        #self.lu1 = sla.splu(self.U.T)
        #self.lu2 = sla.splu(self.L.T)
        #AdjLambda = self.PR.T @ self.lu2.solve(self.lu1.solve(self.PC.T @ (-AdjRHS)))
        #AdjLambda  = self.PR.T @  scipy.sparse.linalg.spsolve(self.L.T, scipy.sparse.linalg.spsolve(self.U.T, self.PC.T @ (-AdjRHS)))
        AdjLambda  = self.PR.T @  spsolve(self.L.T, spsolve(self.U.T, self.PC.T @ (-AdjRHS)))
        #print(np.shape(AdjRHS))
        #raise()
        #plt.imshow(np.reshape(np.real(AdjLambda), (self.nodesY, self.nodesX)))
        #plt.show()
        #raise()
        

        # -----------------------------------------------------------------------------------
        #  COMPUTE SENSITIVITIES 
        # -----------------------------------------------------------------------------------
        self.sens = self.compute_sensitivities(self.LEM, self.dAdx, self.T, AdjLambda)

        
        # -----------------------------------------------------------------------------------
        #  FILTER  SENSITIVITIES 
        # -----------------------------------------------------------------------------------

        DdFSTDFS = filThr.deriv_threshold(self.dFPS)
        sensFOM = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), self.sens,DdFSTDFS)
        #sensFOM_F = filThr.density_filter(filThr.filSca, np.ones((self.nEly,self.nElx), dtype="complex128"), self.sens_F,DdFSTDFS_F)
        #sensFOM = sensFOM  - sensFOM_F
        # -----------------------------------------------------------------------------------
        #  SENSITIVITIES FOR DESIGN REGION
        # -----------------------------------------------------------------------------------
        #sensFOM = sensFOM.flatten()

        #for i in range(len(self.edofMat)):

        #    dFdx_e =  np.array([self.dFdx[n] for n in self.edofMat[i].astype(int)]).flatten()
        #    AdjLambda_e = np.array([AdjLambda[n] for n in self.edofMat[i].astype(int)])
        #    sensFOM [i] -= np.real((AdjLambda_e[np.newaxis] @ dFdx_e)) [0]

        #sensFOM = np.reshape(sensFOM, (self.nEly, self.nElx))

        sensFOM = sensFOM[indexes_design_region[0,:], indexes_design_region[1,:]] 
        
        #sensFOM = self.sens[indexes_design_region[0,:], indexes_design_region[1,:]] 

        # -----------------------------------------------------------------------------------
        #  FOM FOR MINIMIZATION
        # -----------------------------------------------------------------------------------

        FOM =  FOM
        sensFOM =  sensFOM 

        #if self.sign:
        #    sensFOM =  -sensFOM

        # Plotting and printing per optimization iteration
        end = time.time()
        elapsed_time = [(end - start) // 60, end - start - (end - start) // 60 * 60]
        #print("Elapsed time in iteration: "+str(int(elapsed_time[0]))+" min "+str(int(round(elapsed_time[1],0)))+" s")
        #print("----------------------------------------------")
        #plot_iteration(self)

        return FOM, sensFOM

        