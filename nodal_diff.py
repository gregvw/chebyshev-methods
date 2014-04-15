import numpy as np

def nodal_diff(u):
    n = len(u)-1
    s = np.sin(np.pi*np.arange(1,n)/n)
    U = np.hstack((u,np.flipud(u[1:-1])))
    Uhat = np.real(np.fft.fft(U))
    k = np.roll(np.arange(1-n,n+1),1-n)
    What = 1j*k*Uhat
    What[n] = 0
    W = np.real(np.fft.ifft(What))
    w = np.zeros(n+1)
    w[1:-1] = -W[1:n]/s
    i = np.arange(n)
    i2 = i**2
    w[0] = np.dot(i2,Uhat[:n])/n + 0.5*n*Uhat[n]
    w[n] = np.dot(((-1)**(i+1))*i2,Uhat[:n])/n +\
           0.5*(-1)**(n+1)*n*Uhat[n]
    return w 
   
if __name__ == '__main__':
    np.set_printoptions(precision=16,linewidth=99)
    n = 128
    x = np.cos(np.pi*np.arange(n+1)/n)
    f = np.exp(-2*x)

    fp = nodal_diff(f)
    print(fp/f)

