from MMA import mmasub, asymp, gcmmasub, concheck, raaupdate, kktcheck
import numpy as np


class optimizer:

    def __init__(self,
                 m,
                 n,
                 p,
                 xmin,
                 xmax,
                 f0, 
                 f,
                 f_no_d,
                 f_c,
                 f_no_c,
                 maxiter_cont_l,
                 a0,
                 a, 
                 c,
                 d,
                 a_c,
                 c_c,
                 d_c,
                 maxiter,
                 move,
                 maxiniter = 0,
                 type_MMA = "MMA",
                 logfile = False,
                 directory = None,
                 ):

        """Initializing optimizer.
        @ m : Number of constraints.
        @ n : Number of variables. 
        @ x : Input variables (guess).
        m
        """

        self.m = m
        self.n = n
        self.p = p # number of objective functions to be used in minmax
        self.xmin = xmin
        self.xmax = xmax
        self.f0 = f0 # objective function
        self.f = f # constraint function
        self.f_c = f_c # constraint function after lentghscale continuation
        self.f_no_d = f_no_d # constraint function
        self.f_no_c = f_no_c # constraint function lengthscale continuation no derivative
        self.a0 =  a0
        self.a = a
        self.c = c
        self.d = d
        self.a_c = a_c
        self.c_c = c_c
        self.d_c = d_c
        self.maxiter_cont_l = maxiter_cont_l
        self.maxiter = maxiter
        self.move = move
        self.maxiniter = maxiniter
        self.type_MMA = type_MMA
        self.FOM_it = np.zeros(self.maxiter)
        self.cons_1_it = np.zeros(self.maxiter)
        self.cons_2_it = np.zeros(self.maxiter)
        self.cons_3_it = np.zeros(self.maxiter)
        self.cons_4_it = np.zeros(self.maxiter)
        self.cons_5_it = np.zeros(self.maxiter)
        self.lam_array = np.zeros((m, self.maxiter))
        self.logfile = logfile
        self.dir = directory


    def optimize(self, x, xold1 = None,  xold2 = None, low=None, upp = None, raa0 = None, raa = None, outit=0):

        xval = x # initial guess 
        if not hasattr(xold1, "__len__"):
            xold1 = x
        if not hasattr(xold2, "__len__"):
            xold2 = x 

        fval = np.zeros(self.m)[:,np.newaxis]
        fvalnew = np.zeros(self.m)[:,np.newaxis]
        dfdx = np.zeros((self.m, self.n))
        dfdxnew = np.zeros((self.m, self.n))

        if not hasattr(low, "__len__"):
            low = np.zeros(self.n)[:,np.newaxis]
        if not hasattr(upp, "__len__"):
            upp = np.zeros(self.n)[:,np.newaxis]

        if self.type_MMA == "GCMMA":

            epsimin = 1E-7
            if not hasattr(raa0, "__len__"):
                raa0 = 0.01

            #eeen = np.ones((self.n,1))
            eeem = np.ones((self.m,1))

            if not hasattr(raa, "__len__"):
                raa = 0.01*eeem

            raa0eps = 0.000001
            raaeps = 0.000001*eeem
            kkttol = 0
            kktnorm = kkttol+10
            #outit = 0
            outeriter = outit
            xmma = xval

            counter_l = 0

            f0val, df0dx = self.f0(xval, outit)

            self.FOM_it [outit]= f0val

            if self.f == np.array([]):
                fval = np.zeros(1)
                dfdx = np.zeros((1, self.n))

            else:
                if counter_l == 0:
                    for j in range(len(self.f)): 
                        fval[j], dfdx[j, :] = self.f[j](xval)
                        if j == 0:
                            self.cons_1_it [outit]= 40 -fval [j]
                        if j == 1: 
                            self.cons_2_it [outit]= 40 -fval [j]
                        if j == 2: 
                            self.cons_3_it [outit]= fval [j]

                else: 
                    for j in range(len(self.f)): 
                        fval[j], dfdx[j, :] = self.f[j](xval)
                        if j == 0:
                            self.cons_1_it [outit]= 40 -fval [j]
                        if j == 1: 
                            self.cons_2_it [outit]= 40 -fval [j]
                        if j == 2: 
                            self.cons_3_it [outit]= fval [j]
                        if j == 3: 
                            self.cons_4_it [outit]= fval [j]
                        if j == 4: 
                            self.cons_5_it [outit]= fval [j]


            while (kktnorm > kkttol) and (outit < self.maxiter-1):

                print("----------------------------------------------")
                print("Optimization iteration: ",outit)

                if outit > self.maxiter_cont_l and counter_l == 0:

                    self.f = self.f_c
                    self.f_no_d = self.f_no_c
                    self.m = self.m + 2 # 2 lengthscale constraints
                    fval = np.zeros(self.m)[:,np.newaxis]
                    fvalnew = np.zeros(self.m)[:,np.newaxis]
                    dfdx = np.zeros((self.m, self.n))
                    dfdxnew = np.zeros((self.m, self.n))
                    for j in range(len(self.f)): 
                        fval[j], dfdx[j, :] = self.f[j](xval)

                    eeem = np.ones((2,1))
                    raa_c = 0.01*eeem
                    raa = np.concatenate([raa, raa_c])
                    raaeps_c = 0.000001*eeem
                    raaeps = np.concatenate([raaeps, raaeps_c])

                    eeem = np.ones((self.m,1))

                    self.a = self.a_c
                    self.c = self.c_c
                    self.d = self.d_c
                    
                    counter_l += 1
                
                outit += 1
                outeriter += 1
                # The parameters low, upp, raa0 and raa are calculated:
                low,upp,raa0,raa= asymp(outeriter,self.n,xval,xold1,xold2,self.xmin,self.xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx,dfdx)

                # The MMA subproblem is solved at the point xval:
                xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp= gcmmasub(self.m,self.n,iter,epsimin,xval,self.xmin,self.xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,self.a0,self.a,self.c,self.d, self.move)
                # The user should now calculate function values (no gradients) of the objective- and constraint
                # functions at the point xmma ( = the optimal solution of the subproblem).

                f0valnew ,_ = self.f0(xmma, outit) #no gradient

                if self.f == np.array([]):
                    fvalnew = np.zeros(1)
                    dfdxnew = np.zeros((1, self.n))

                else:
                    for j in range(len(self.f)): 
                        fvalnew[j]= self.f_no_d[j](xmma) # no gradient
                

                # It is checked if the approximations are conservative:
                conserv = concheck(self.m,epsimin,f0app,np.array([f0valnew]),fapp,fvalnew)

                # While the approximations are non-conservative (conserv=0), repeated inner iterations are made:
                innerit = 0
                if conserv == 0:
                    while conserv == 0 and innerit <= self.maxiniter:
                        innerit += 1
                        # New values on the parameters raa0 and raa are calculated:
                        raa0,raa = raaupdate(xmma,xval,self.xmin,self.xmax,low,upp,f0valnew,fvalnew,f0app,fapp,raa0,raa,raa0eps,raaeps,epsimin)
                        # The GCMMA subproblem is solved with these new raa0 and raa:
                        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(self.m,self.n,iter,epsimin,xval,self.xmin, self.xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,self.a0,self.a,self.c,self.d, self.move)
                        # The user should now calculate function values (no gradients) of the objective- and 
                        # constraint functions at the point xmma ( = the optimal solution of the subproblem).
                        f0valnew ,_ = self.f0(xmma, outit) # no gradient

                        if self.f_no_d == np.array([]):
                            fvalnew = np.zeros(1)

                        else:
                            for j in range(len(self.f_no_d)): 
                                fvalnew[j] = self.f_no_d[j](xmma) # no gradient

                        # It is checked if the approximations have become conservative:
                        conserv = concheck(self.m,epsimin,f0app,np.array([f0valnew]),fapp,fvalnew)


                xold2 = xold1.copy()
                xold1 = xval.copy()
                xval = xmma.copy()
                
                f0val, df0dx = self.f0(xval, outit, plot=True)

                self.FOM_it [outit]= f0val

                if self.f == np.array([]):
                    fval = np.zeros(1)
                    dfdx = np.zeros((1, self.n))

                else:
                    if counter_l == 0:
                        for j in range(len(self.f)): 
                            fval[j], dfdx[j, :] = self.f[j](xval)
                            if j == 0:
                                self.cons_1_it [outit]= 40 -fval [j]
                            if j == 1: 
                                self.cons_2_it [outit]= 40 -fval [j]
                            if j == 2: 
                                self.cons_3_it [outit]= fval [j]

                    else: 
                        for j in range(len(self.f)): 
                            fval[j], dfdx[j, :] = self.f[j](xval)
                            if j == 0:
                                self.cons_1_it [outit]= 40 -fval [j]
                            if j == 1: 
                                self.cons_2_it [outit]= 40 -fval [j]
                            if j == 2: 
                                self.cons_3_it [outit]= fval [j]
                            if j == 3: 
                               self.cons_4_it [outit]= fval [j]
                            if j == 4: 
                               self.cons_5_it [outit]= fval [j]

                residu,kktnorm,residumax = kktcheck(self.m,self.n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,self.xmin,self.xmax,df0dx,fval,dfdx,self.a0,self.a,self.c,self.d)

                self.lam_array[:3, outit] = lam.flatten()[:3]



        if self.logfile:
            import os
            directory_opt = self.dir+"/optimization_parameters"
            if not os.path.exists(directory_opt):
                    os.makedirs(directory_opt)
            np.save(self.dir+"/optimization_parameters/"+"xold1.npy", xold1)
            np.save(self.dir+"/optimization_parameters/"+"xold2.npy", xold2)
            np.save(self.dir+"/optimization_parameters/"+"low.npy", low)
            np.save(self.dir+"/optimization_parameters/"+"upp.npy", upp)
            np.save(self.dir+"/optimization_parameters/"+"raa0.npy", raa0)
            np.save(self.dir+"/optimization_parameters/"+"raa.npy", raa)
            


        elif self.type_MMA == "MMA":
            
            for i in range(self.maxiter):

                print("----------------------------------------------")
                print("Optimization iteration: ",i)

                f0val, df0dx = self.f0(xval,i)
                self.FOM_it [i]= f0val

                if self.f == np.array([]):
                    fval = np.zeros(1)
                    dfdx = np.zeros((1, self.n))

                else:
                    for j in range(len(self.f)): 
                        fval[j], dfdx[j, :] = self.f[j](xval)
                        if j == 0:
                            self.cons_1_it [i]= 40 - fval [j]
                        if j == 1: 
                            self.cons_2_it [i]= 40 - fval [j]

                xval_new, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(self.m,self.n,i,xval,self.xmin, self.xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,self.a0,self.a,self.c,self.d,self.move)
            
                xold2 = xold1
                xold1 = xval
                xval = xval_new

                self.lam_array[:, i] = lam.flatten()


        return xval, self.FOM_it, self.cons_1_it, self.cons_2_it, self.cons_3_it, self.cons_4_it, self.cons_5_it






