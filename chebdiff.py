import numpy as np
import scipy as sp

def modalDiffMatrix(n):
    """Return the modal differentiation matrix for 
       Chebyshev coefficients"""
    k = np.arange(n)
    a = (-1)**k
    A = sp.triu(1-np.outer(a,a))
    D = np.dot(A,np.diag(k))
    D[0,:] = D[0,:]/2
    return D  
    
    
def modalDiff(A):
    """Differentiate a set of Chebyshev polynomial expansion 
	 coefficients"""
    size = A.shape
    if len(size) == 2:
        m,n = size
        SA = A*np.outer(2*np.arange(m),np.ones(n))
        DA = np.zeros((m,n))
        DA[m-3:m-1,:]=SA[m-2:m,:]	  
        for j in np.arange(np.floor(m/2)-2):
            k = m-3-2*j
            DA[k,:] = SA[k+1,:] + DA[k+2,:]
            DA[k-1,:] = SA[k,:] + DA[k+1,:]
  
        DA[1,:] =  SA[2,:] + DA[3,:]
        DA[0,:] = (SA[1,:] + DA[2,:])*0.5
    else:
        m = size[0]
        SA = A*2*np.arange(m)
        DA = np.zeros(m)
        DA[m-3:m-1]=SA[m-2:m]	  
        for j in np.arange(np.floor(m/2)-2):
            k = m-3-2*j
            DA[k] = SA[k+1] + DA[k+2]
            DA[k-1] = SA[k] + DA[k+1]
  
        DA[1] =  SA[2] + DA[3]
        DA[0] = (SA[1] + DA[2])*0.5

    return DA