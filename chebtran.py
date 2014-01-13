# chebtran.py 

import numpy as np
from scipy.fftpack import fft, ifft, dct, idct


def fcgt(A): # Modal Coefficients to Gauss Nodal
    """
    Fast Chebyshev-Gauss transformation from 
    Chebyshev expansion coefficients (modal) to point 
    space values (nodal). If I=numpy.identity(n), then
    T=chebtran.fcgt(I) will be the Chebyshev 
    Vandermonde matrix on the Gauss nodes
    """
    from copy import copy
    size = A.shape
    Acopy = copy(A) # Shallow copy
    if len(size) == 2:       # Multiple vectors
        Acopy[1:,:] /= 2  
        B = dct(Acopy,type=3,axis=0)
    else:                  # Single Vector
        Acopy[1:] /= 2 
        B = dct(Acopy,type=3)

    return B


def ifcgt(A):  # Gauss Nodal to Modal Coefficients
    """
    Fast Chebyshev-Gauss transformation from 
    point space values (nodal) to Chebyshev expansion 
    coefficients (modal). If I=numpy.identity(n), then
    Ti=chebyshev.ifcglt(I) will be the inverse of the 
    Chebyshev Vandermonde matrix on the Gauss nodes
    """
    size = A.shape
    B = idct(A,3,axis=0,norm=None)
    if len(size) == 2:      # Multiple vectors
        B[0,:] /= 2*size[0]
        B[1:,:] /= size[0]
    else:                   # Single Vector
        B[0] /= 2*size[0]
        B[1:] /= size[0]

    return B 


def fcglt(A): # Modal Coefficients to Lobatto Nodal
    """
    Fast Chebyshev-Gauss-Lobatto transformation from 
    Chebyshev expansion coefficients (modal) to point 
    space values (nodal). If I=numpy.identity(n), then
    T=chebyshev.fcglt(I) will be the Chebyshev 
    Vandermonde matrix on the Lobatto nodes
    """
    size = A.shape
    m = size[0]
    k = m-2-np.arange(m-2)

    if len(size) == 2: # Multiple vectors
        V = np.vstack((2*A[0,:],A[1:m-1,:],2*A[m-1,:],A[k,:]))
        F = fft(V, n=None, axis=0)
        B = 0.5*F[0:m,:]
    else:  # Single vector
        V = np.hstack((2*A[0],A[1:m-1],2*A[m-1],A[k]))
        F = fft(V, n=None)
        B = 0.5*F[0:m]

    if A.dtype!='complex':
        return np.real(B)
    else:
        return B
    
def ifcglt(A): # Lobatto Nodal to Modal Coefficients
    """
    Fast Chebyshev-Gauss-Lobatto transformation from 
    point space values (nodal) to Chebyshev expansion 
    coefficients (modal). If I=numpy.identity(n), then
    Ti=chebyshev.ifcglt(I) will be the inverse of the 
    Chebyshev Vandermonde matrix on the Lobatto nodes
    """
    size = A.shape
    m = size[0]
    k = m-1-np.arange(m-1)
  
    if len(size) == 2: # Multiple vectors
        V = np.vstack((A[0:m-1,:],A[k,:]))
        F = ifft(V, n=None, axis=0)
        B = np.vstack((F[0,:],2*F[1:m-1,:],F[m-1,:]))
    else:  # Single vector
        V = np.hstack((A[0:m-1],A[k]))
        F = ifft(V, n=None)
        B = np.hstack((F[0],2*F[1:m-1],F[m-1]))
        
    if A.dtype!='complex':
        return np.real(B)
    else:
        return B


  
  
    
  