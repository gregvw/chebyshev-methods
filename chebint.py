import numpy as np
from scipy.linalg import solve_banded
from chebtran import fcgt, ifcgt

class integrator(object):
    
    def __init__(self,n):
        # Store LU factors of K matrix in band form
        L0 = np.ones(n)
        L1 = -np.hstack((0.5,np.ones(n-1)))
        self.Lbands = np.vstack((L0,L1))

        U0 = np.hstack((4,np.arange(2,n+1)))*0.5
        U1 = np.hstack((0,np.arange(n-1)))*0.5
        self.Ubands = np.vstack((U1,U0))

        k = np.arange(n)    
        theta = np.pi*(2*k+1)/(2*n)
        self.x = np.cos(theta) 

    def getx(self):
        return self.x

    def solve(self,f): # Solves u'=f for u(-1)=0
        fhat = ifcgt(f*(1-self.x)) # Compute (psi_j,f)
        fhat[0] *= 2
        y = solve_banded((1,0),self.Lbands,fhat,\
            overwrite_ab=False,overwrite_b=True)
        uhat = solve_banded((0,1),self.Ubands,y,\
            overwrite_ab=False,overwrite_b=True)
        u = fcgt(uhat)*(1+self.x) # u=phi*uhat
        return u