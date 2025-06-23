from element_matrices import element_matrices
from material_interpolation import material_interpolation_metal
import scipy
from scipy.sparse.linalg import use_solver
import numpy as np
from element_matrices import evaluate_edge_at_el
from slepc4py import SLEPc
from petsc4py import PETSc

class dis:
    "Class that describes the discretized FEM model"
    def __init__(self, 
                 scaling,
                 nElx,
                 nEly):
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
        

        self.iS = np.reshape(np.kron(self.edofMat,np.ones((8,1))), 64*self.nElx*self.nEly) # nodes in one direction
        self.iS_node = np.reshape(np.kron(self.edofMat,np.ones((4,1))), 32*self.nElx*self.nEly) # nodes in one direction
        self.jS = np.reshape(np.kron(self.edofMat,np.ones((1,8))), 64*self.nElx*self.nEly)  # nodes in the other direction
        self.jS_node = np.reshape(np.kron(self.edofMat,np.ones((1,4))), 32*self.nElx*self.nEly) # nodes in one direction
        

        edges_per_element = np.reshape(np.repeat(elem_nrs_flat, 4),(nEX*nEY,4))  + np.reshape(np.repeat(np.repeat(np.linspace(0,nEY-1,nEY) * nEX,nEX),4), (nEX*nEY,4)) + np.ones((nEX*nEY,4))*np.tile(np.array([nEX+(nEX+1), 0, nEX, nEX+1]), (nEX*nEY, 1)) 
        self.edges_per_element = edges_per_element

        self.edofMat_total = np.hstack([self.edofMat, self.edges_per_element]) # we get all the DOFs per element

        self.iS_edge = np.reshape(np.kron(edges_per_element,np.ones((4,1))), 16*self.nElx*self.nEly)    # edges in one direction
        self.iS_edge_node = np.reshape(np.kron(edges_per_element,np.ones((8,1))), 32*self.nElx*self.nEly)    # edges in one direction

        self.jS_edge = np.reshape(np.kron(edges_per_element,np.ones((1,4))), 16*self.nElx*self.nEly)   # edges in the other direction
        self.jS_edge_node = np.reshape(np.kron(edges_per_element,np.ones((1,8))), 32*self.nElx*self.nEly)    # edges in one direction

        
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

        self.nBC = np.unique(np.concatenate([self.n1BC_Ez, self.n2BC_Ez, self.n2BC_Ez_2, self.n3BC_Ez, self.n3BC_Ez_2, self.n4BC_Ez]))
        self.nBC_edge = np.unique(np.concatenate([self.n1BC_edge, self.n2BC_edge, self.n3BC_edge, self.n4BC_edge]))
        #self.nBC = np.concatenate([self.n1BC_edge, self.n2BC_edge, self.n3BC_edge, self.n4BC_edge])
        self.nBC = np.concatenate([self.nBC, self.nBC_edge]) # we concatenate the boundary conditions for the nodes and edges

        # -----------------------------------------------------------------------------------
        # C) SET INDEXES FOR THE FULL NODE MATRIX
        # ----------------------------------------------------------------------------------- 
        
        # to match all elements with nodes (and vice versa) we flatten the DOF matrix
        self.edofMatfull = np.hstack([self.edofMat, self.edges_per_element])
        
        

    
    def material_distribution(self, dVs):
        """
        Sets the material in the simulation domain.
        In this case, we set the air region, the substrate and the deisgn region.
        @ dVs: Values for the design variables.
        """ 

        dFP = np.zeros((self.nEly, self.nElx), dtype="complex128") # we initialize the domain to the background material
        dFP = np.reshape(dVs, np.shape(dFP))
        #limit1y = int(0.15 * self.nEly)
        #limit2y = int(0.85 * self.nEly)
        #limit1x = int(0.15 * self.nElx)
        #limit2x = int(0.85 * self.nElx)

        #dFP[:limit1y, :] = 1.0 
        #dFP[limit2y:, :] = 1.0
        #dFP[:, :limit1x] = 1.0
        #dFP[:, limit2x:] = 1.0
        
        return dFP
    
    def assemble_matrix(self, eps, phy):
        """
        Assembles the global system matrix.
        Since all our elements are linear rectangular elements we can use the same element matrices for all of the elements.
        @ eps: Value for the dielectric constant for all elements.
        @ phy: physics class objects that holds the physical parameters of the system..
        """ 


        eps_S_4 = np.repeat(eps.flatten(),16)
        eps_S_8 = np.repeat(eps.flatten(),64)


        self.ATT_S = np.tile(self.E_mat.flatten(),self.nElx*self.nEly) - phy.k**2*eps_S_4*np.tile(self.F_mat.flatten(),self.nElx*self.nEly)
        self.BTT_S = np.tile(self.F_mat.flatten(),self.nElx*self.nEly)
        self.BZT_S = np.tile(self.BZT.flatten(),self.nElx*self.nEly)
        self.BTZ_S = np.tile(self.BTZ.flatten(),self.nElx*self.nEly)
        self.BZZ_S = np.tile(self.BZZ1.flatten(),self.nElx*self.nEly) - phy.k**2*eps_S_8*np.tile(self.BZZ2.flatten(),self.nElx*self.nEly)

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
    

        boundary_values = np.ones_like(self.nBC)
        I = scipy.sparse.identity(n=self.num_dof, format ="csr", dtype='complex128')

        N = I - scipy.sparse.csr_matrix((boundary_values,(self.nBC.astype(int), self.nBC.astype(int))), shape=(self.num_dof,self.num_dof), dtype='complex128')

        S = N.T @ S @ N + (I - N)

        M = N.T @ M @ N 
        
        return S.tocsr(), M.tocsr()
    
    def solve_eigenproblem(self, phy, use_slepc=True):
        """
        Solves the eigenproblem using either scipy or SLEPc based on the flag.
        Filters out eigenvalues and corresponding eigenvectors where the eigenvalue is close to zero.
        @ phy: Physics class object that holds the physical parameters of the system.
        @ use_slepc: Boolean flag to use SLEPc solver if True, otherwise use scipy.eigs.
        """
        sigma = (phy.k) ** 2  # shift for the eigenvalue problem, can be modified to calculate more eigenvalues
        print('Solving eigenvalue problem...')
        self.eig_num_tot = 15  # can be modified to calculate more eigenvalues

        if use_slepc:
            try:
                print("Using SLEPc solver...")
                # Convert matrices to PETSc format
                S_petsc = PETSc.Mat().createAIJ(size=self.S.shape, csr=(self.S.indptr, self.S.indices, self.S.data))
                M_petsc = PETSc.Mat().createAIJ(size=self.M.shape, csr=(self.M.indptr, self.M.indices, self.M.data))

                # Set up the SLEPc eigenproblem solver
                eps = SLEPc.EPS().create()
                eps.setOperators(S_petsc, M_petsc)
                eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Generalized Non-Hermitian EVP

                # Set solver parameters
                eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # Find eigenvalues near target
                eps.setTarget(-sigma)  # sigma: complex spectral shift
                eps.setDimensions(nev=self.eig_num_tot)  # Number of eigenvalues to compute
                eps.setTolerances(tol=1e-10, max_it=1500)

                # Use shift-and-invert spectral transformation
                st = eps.getST()
                st.setType(SLEPc.ST.Type.SINVERT)
                st.setShift(sigma)

                eps.solve()

                nconv = eps.getConverged()
                if nconv > 0:
                    eigval = []
                    eigvec = []
                    j = 0  # Counter for eigenvalues that fulfill the condition
                    for i in range(min(nconv, self.eig_num_tot)):
                        vr, vi = S_petsc.getVecs()
                        eigval_i = eps.getEigenpair(i, vr, vi)
                        wavevector_i = np.sqrt(-eigval_i)
                        eff_index_i = wavevector_i / phy.k  
                        if abs(eigval_i) > 1e-6:  # Filter out eigenvalues close to zero
                            eigval.append(eigval_i)
                            print(f"Effective index {j}: {eff_index_i.real:.4f} + {eff_index_i.imag:.4f}j")
                            eigvec.append(vr.getArray())
                            j += 1  # Increment the counter
                    return np.array(eigval), np.array(eigvec).T
                else:
                    raise RuntimeError("No eigenvalues converged.")
            except ImportError:
                raise ImportError("SLEPc or PETSc is not installed. Please install them to use this solver.")
        else:
            print("Using scipy.eigs solver...")
            eigval, eigvec = scipy.sparse.linalg.eigs(A=self.S, k=self.eig_num_tot, M=self.M, sigma=-sigma)
            mask = np.abs(eigval) > 1e-6  # Filter out eigenvalues close to zero
            return eigval[mask], eigvec[:, mask]
        
    def get_lu_factorization_matrices(self):
        """
        Gives the LU factorization of a sparse matrix.
        Definitions from reference (scipy.sparse.linalg.SuperLU documentation), adjusted to case.
        """ 
        L = self.lu.L
        U = self.lu.U
        PR = scipy.sparse.csr_matrix((np.ones(self.num_dof, dtype="complex128"), (self.lu.perm_r, np.arange(self.num_dof))), dtype="complex128") # Row permutation matrix
        PC = scipy.sparse.csr_matrix((np.ones(self.num_dof, dtype="complex128"), (np.arange(self.num_dof), self.lu.perm_c)), dtype="complex128") # Column permutation matrix
    
        return L, U, PR, PC
    
    def compute_sensitivities(self, phy, dAdx, E, AdjLambda):
        """
        Computes the sensitivities for all of the elements in the simulation domain.
        @ k: wavevector of the problem (Frequency domain solver).
        @ dAdx: derivative of the design variables in the simulation domain. 
        @ E: electric field calculated from the forward problem.
        """ 

        sens = np.zeros(self.nEly * self.nElx, dtype="complex128")


        for i in range(len(self.edofMat_total)):


            dAtt_e =  -phy.k**2 * np.reshape(self.F_mat*dAdx.flatten() [i], (4,4))
            dBzz_e =  -phy.k**2 * np.reshape(self.BZZ2*dAdx.flatten() [i], (8,8))

            dSdx_e = np.zeros((12,12), dtype="complex128")
            dMdx_e = np.zeros((12,12), dtype="complex128")


            dMdx_e [:8,:8] =  dBzz_e
            dSdx_e [8:,8:] =  dAtt_e 

            sens [i] = 0 # TO BE FINISHED

    
        return np.reshape(sens, (self.nEly, self.nElx))


    
    def FEM_sol(self, dVs,  phy, filThr):
        """
        Gives the solution to the forward FEM problem; this is, the electric field solution.
        @ dVs: Design variables in the simulation domain
        @ phy: Physics class objects that holds the physical parameters of the system
        @ filThr: Filtering and thresholding class object
        @ solver: either the RHS or eigenvalue solver.
        @ eigval_num : Eiegenvalue to be selected out of all the calculated.
        """ 
        # -----------------------------------------------------------------------------------
        # FILTERING AND THRESHOLDING ON THE MATERIAL
        # ----------------------------------------------------------------------------------- 

        self.dFP = self.material_distribution(dVs)
        self.dFPS = filThr.density_filter(np.ones((self.nEly, self.nElx), dtype="complex128"), filThr.filSca, self.dFP, np.ones((self.nEly, self.nElx), dtype="complex128"))
        self.dFPST = filThr.threshold(self.dFPS)
  
        # -----------------------------------------------------------------------------------
        # MATERIAL INTERPOLATION
        # ----------------------------------------------------------------------------------- 
        
        self.eps, self.depsdx = material_interpolation_metal(phy.n_metal, phy.k_metal, phy.n_back, phy.k_back, self.dFPST, phy.alpha) 

        # -----------------------------------------------------------------------------------
        # ASSEMBLY OF GLOBAL SYSTEM MATRIX
        # -----------------------------------------------------------------------------------
        self.S, self.M = self.assemble_matrix(self.eps, phy)

        # -----------------------------------------------------------------------------------
        # SOLVE SYSTEM OF EQUATIONS
        # -----------------------------------------------------------------------------------

        self.eigvals, self.eigvecs = self.solve_eigenproblem(phy)

        return self.eigvals, self.eigvecs

    def get_fields(self, phy, eigval_num=0):

        self.E = self.eigvecs[:, eigval_num]

        kz = np.sqrt(-self.eigvals[eigval_num])  
            
        self.Et = self.E[:self.N_edges]/kz
            
        indexes_nodes_4 = np.tile(np.arange(self.nodesX_new)[::2], self.nEly+1) + np.repeat((self.nodesX_new+self.nodesY_new)*np.arange(self.nEly+1),self.nElx+1) + self.N_edges

        self.Ez =  self.E[indexes_nodes_4] / (-1j)
            
        from element_matrices import evaluate_edge_at_el
        Et_edges = self.E[:self.N_edges]/kz
        self.Ex = np.zeros(self.nodesX*self.nodesY, dtype='complex128')
        self.Ey = np.zeros(self.nodesX*self.nodesY, dtype='complex128')
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
            
        self.eigenval_kz = kz/phy.k

        
        self.normE = np.sqrt(self.Ex.flatten()*np.conj(self.Ex.flatten())+self.Ey.flatten()*np.conj(self.Ey.flatten())+self.Ez.flatten()*np.conj(self.Ez.flatten()))

        return self.E







