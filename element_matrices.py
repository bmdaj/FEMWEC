import numpy as np
from sympy import *
#from IPython.display import display

def element_matrices(scaling):
    """
    Calculates the element matrices used to construct the global system matrix.
    In our case we assume linear rectangular elements with local indexing:
    1 2 
    3 4.
    @ scaling: size scale of the problem, i.e. 1e-9 we work in nanometers.
    """

    a = scaling / 2 # Element size scaling
    b = scaling / 2 # Element size scaling

    def ip_quad_2o(a,b): 
        """
        Implementation for the interpolation functions of first order for linear rectangular elements.
        In our case we assume linear rectangular elements with local indexing:
        7 6 5
        8 4
        1 2 3
        @ a: side of the rectangle in the X direction.
        @ b: side of the rectangle in the Y direction.
        """

        N_1 = lambda x,y:   -0.25 * (1-x/a) * (1-y/b) * (1+x/a+y/b)  
        N_2 = lambda x,y:    0.5 * (1-x/a) * (1+x/a) * (1-y/b) 
        N_3 = lambda x,y:   -0.25 * (1+x/a) * (1-y/b) * (1-x/a+y/b)
        N_4 = lambda x,y:    0.5 * (1+x/a) * (1+y/b) * (1-y/b) 
        N_5 = lambda x,y:   -0.25 * (1+x/a) * (1+y/b) * (1-x/a-y/b)
        N_6 = lambda x,y:   0.5 * (1-x/a) * (1+x/a) * (1+y/b) 
        N_7 = lambda x,y:  -0.25 * (1-x/a) * (1+y/b) * (1+x/a-y/b)
        N_8 = lambda x,y:   0.5 * (1-x/a) * (1+y/b) * (1-y/b) 

        return N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8


    def ip_edge_1o (a):

        N_1 = lambda x,y: (1/(2*a)) * np.array([ a - y, 0.0])
        N_2 = lambda x,y: (1/(2*a)) * np.array([ a + y, 0.0])
        N_3 = lambda x,y: (1/(2*a)) * np.array([0.0, a - x])
        N_4 = lambda x,y: (1/(2*a)) * np.array([0.0, a + x])

        return N_1, N_2, N_3, N_4


    def ip_edge_1o_dx (a):

        N_1 = lambda x,y: (1/(2*a)) * np.array([0.0, 0.0])
        N_2 = lambda x,y: (1/(2*a)) * np.array([0.0, 0.0])
        N_3 = lambda x,y: (1/(2*a)) * np.array([0.0, -1.0])
        N_4 = lambda x,y: (1/(2*a)) * np.array([0.0, 1.0])

        return N_1, N_2, N_3, N_4

    def ip_edge_1o_dy (a):

        N_1 = lambda x,y: (1/(2*a)) * np.array([-1.0, 0.0])
        N_2 = lambda x,y: (1/(2*a)) * np.array([1.0, 0.0])
        N_3 = lambda x,y: (1/(2*a)) * np.array([0.0, 0.0])
        N_4 = lambda x,y: (1/(2*a)) * np.array([0.0, 0.0])

        return N_1, N_2, N_3, N_4

    
    
    def calculate_integral(f, a, b):
        """
        Function to calculate surface integrals in two dimensional spaces.
        @
        @ a: side of the rectangle in the X direction (i.e. start/end of integration domain).
        @ b: side of the rectangle in the Y direction (i.e. start/end of integration domain).
        """

        import scipy.integrate as integrate
        return integrate.dblquad(f, -a, a, -b, b)


    # -----------------------------------------------------------------------------------
    # CACULATION OF THE A AND A' MATRICES
    # ----------------------------------------------------------------------------------- 

    ipf = list(ip_quad_2o(a,b)) # initialize the interpolation functions

    
    ipf_dx = []
    ipf_dy = []

    x, y = symbols('x y')

    for i in range(8):

        ipf_dx.append(lambdify((x,y), ipf[i](x,y).diff(x)))
        ipf_dy.append(lambdify((x,y), ipf[i](x,y).diff(y)))

    

    ipf_edge = list(ip_edge_1o(a))


    ipf_edge_dx = list(ip_edge_1o_dx(a))
    ipf_edge_dy = list(ip_edge_1o_dy(a))

    E = np.ones((4,4), dtype="complex128") # matrix for rectangular edge elements in Ming
    # CHANGED INDICES WITH NEW LOCAL NUMBERING
    E [0,1] = -1.0 
    E [1,0] = -1.0
    E [0,2] = -1.0 
    E [2,0] = -1.0 
    E [3,1] = -1.0 
    E [1,3] = -1.0 
    E [3,2] = -1.0 
    E [2,3] = -1.0

    F = np.zeros((4,4), dtype="complex128") # matrix for rectangular edge elements in Ming

    fac = (4*a**2) / 6

    F [0,0] = fac * 2
    F [1,1] = fac * 2
    F [2,2] = fac * 2
    F [3,3] = fac * 2
    F [0,1] = fac 
    F [1,0] = fac 
    F [2,3] = fac
    F [3,2] = fac


    E = np.zeros((4,4), dtype="complex128")
    F = np.zeros((4,4), dtype="complex128")
    BZT =  np.zeros((8,4), dtype="complex128")
    BTZ =  np.zeros((4,8), dtype="complex128")
    ZZ1 =  np.zeros((8,8), dtype="complex128")
    ZZ2 = np.zeros((8,8), dtype="complex128")


    for i in range(4):
        for j in range(4):

            integrand_E = lambda x, y : (ipf_edge_dx [i] (x,y) [1] - ipf_edge_dy [i] (x,y) [0]) * (ipf_edge_dx [j] (x,y) [1] - ipf_edge_dy [j] (x,y) [0])
            integrand_F = lambda x, y : np.dot(ipf_edge [i] (x,y), ipf_edge [j] (x,y))

            F [i,j] = calculate_integral(integrand_F, a, b) [0]
            E [i,j] = calculate_integral(integrand_E, a, b) [0]


    for i in range(4):
        for j in range(8):

            integrand_btz = lambda x, y : np.dot(ipf_edge [i] (x,y), np.array([ipf_dx [j] (x,y), ipf_dy [j] (x,y)])) 
            BTZ [i,j] = calculate_integral(integrand_btz, a, b) [0]

    for i in range(8):
        for j in range(4):

            integrand_bzt = lambda x, y : np.dot(np.array([ipf_dx [i] (x,y), ipf_dy [i] (x,y)]), ipf_edge [j] (x,y))
            BZT [i,j] = calculate_integral(integrand_bzt, a, b) [0]

    for i in range(8):
        for j in range(8):

            integrand_bzz1 = lambda x, y : np.dot(np.array([ipf_dx [i] (x,y), ipf_dy [i] (x,y)]), np.array([ipf_dx [j] (x,y), ipf_dy [j] (x,y)])) 
            integrand_bzz2 = lambda x, y : ipf [i] (x,y) * ipf [j] (x,y)
            ZZ1 [i,j] = calculate_integral(integrand_bzz1, a, b) [0]
            ZZ2 [i,j] = calculate_integral(integrand_bzz2, a, b) [0]   



    return E, F, BZT.T, BTZ.T, ZZ1, ZZ2


def ip_edge_1o (a):

    N_1 = lambda x,y: (1/(2*a)) * np.array([ a - y, 0.0])
    N_2 = lambda x,y: (1/(2*a)) * np.array([ a + y, 0.0])
    N_3 = lambda x,y: (1/(2*a)) * np.array([0.0, a - x])
    N_4 = lambda x,y: (1/(2*a)) * np.array([0.0, a + x])

    return N_1, N_2, N_3, N_4

def evaluate_edge_at_el(scaling, ets1, ets2, ets3, ets4):

    a = scaling / 2

    N_1, N_2, N_3, N_4 = ip_edge_1o (a)

    # node 0

    x0 = -a
    y0 = a

    et1 = ets1 * N_1(x0,y0) + ets2 * N_2(x0,y0) + ets3 * N_3(x0,y0) + ets4 * N_4(x0,y0)

    # node 1

    x0 = a
    y0 = a

    et2 = ets1 * N_1(x0,y0) + ets2 * N_2(x0,y0) + ets3 * N_3(x0,y0) + ets4 * N_4(x0,y0)

    # node 2

    x0 = -a
    y0 = -a

    et3 = ets1 * N_1(x0,y0) + ets2 * N_2(x0,y0) + ets3 * N_3(x0,y0) + ets4 * N_4(x0,y0)

    # node 3

    x0 = a
    y0 = -a

    et4 = ets1 * N_1(x0,y0) + ets2 * N_2(x0,y0) + ets3 * N_3(x0,y0) + ets4 * N_4(x0,y0)

    return et1, et2, et3, et4