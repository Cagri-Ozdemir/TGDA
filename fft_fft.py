import numpy as np
from numpy import linalg

def tproddft(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    C = np.zeros((n0,n1,nc), dtype=complex)
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    D = np.fft.fft(A, axis = 0)
    Bhat = np.fft.fft(B, axis = 0)

    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])

    Cx = np.fft.ifft(C, axis=0)
    #Cx = np.real(Cx)

    return Cx

def ttransdft(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b),dtype='complex')
    B[0,:,:] = np.transpose(np.conj((A[0,:,:])))
    for j in range(a-1,0,-1):
        B[a-1-j+1,:,:] = np.transpose(A[j,:,:])
    #B = np.real(B)
    return B

def tinvdft(A):
    (a, b, c) = A.shape
    C = np.zeros((a, b, c), dtype=complex)
    D = np.fft.fft(A, axis=0)
    for j in range(a):
        C[j,:,:]=np.linalg.inv(D[j,:,:])
    C1 = np.fft.ifft(C,axis=0)
    #C1 = np.real(C1)
    return C1



def Class_scatters_dft(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class),dtype="complex")
    Sw =  np.zeros((n1,n2,n2),dtype="complex")
    Sb = np.zeros((n1, n2, n2),dtype="complex")
    a = np.zeros((n1,n2,1),dtype="complex")
    b = np.zeros((n1, n2, 1),dtype="complex")
    Mean_tensor = np.zeros((n1, n2, 1),dtype="complex")
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    for i in range(num_class):
      Sa = np.zeros((n1, n2, n2),dtype="complex")
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      for j in idx:
          a[:,:,0] = Tensor_train[:,:,j]-mean_tensor_train[:,:,i]
          Sa = Sa + tproddft(a,ttransdft(a))
      Sw = Sw+Sa
    for i in range(num_class):
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddft(b,ttransdft(b)))*num_class

    return Sw,Sb

def teigdft(A):
    (n0,n1,n2) = A.shape
    U1 = np.zeros((n0,n1,n2),dtype='complex')
    S1 = np.zeros((n0, n1, n2),dtype='complex')

    arr = np.fft.fft(A, axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      S, U = np.linalg.eig(M)
      np.fill_diagonal(S1[i,:,:],S)
      U1[i, :, :] = U
    U1x = np.fft.ifft(U1,axis=0)
    S1x = np.fft.ifft(S1, axis=0)
    # U1x = np.real(U1x)
    # S1x = np.real(S1x)
    return S1x,U1x

def tSVD(A):
    n0,n1,n2 = A.shape
    U1 = np.zeros((n0,n1,n1),dtype=complex)
    S1 = np.zeros((n0, n1, n2),dtype=complex)
    V1 = np.zeros((n0,n2,n2),dtype=complex)
    A = np.fft.fft(A, axis=0)

    for i in range(n0):
      (U, S, Vt) = np.linalg.svd(A[i,:,:],full_matrices='true')
      np.fill_diagonal(S1[i,:,:],S)
      U1[i,:,:] = U
      Vc = np.conj(Vt)
      V1[i,:,:] = Vc.T

    U1x = (np.fft.ifft(U1, axis=0))
    S1x = (np.fft.ifft(S1, axis=0))
    V1x = (np.fft.ifft(V1, axis=0))


    return U1x, S1x, V1x

def fronorm(A):
    tmp = A*A
    B = np.absolute(tmp)
    C = np.sum(B)
    y = np.sqrt(C)
    return y

def dftdet(A):
    (n0,n1,n2) = A.shape
    D = np.fft.fft(A,axis=0)
    det = np.zeros((n0),dtype="complex")
    for i in range(n0):
        det[i] = np.linalg.det(D[i,:,:])
    D3 = np.fft.ifft(det)
    return D3
#orth1 = u[0,:,:].T@u[0,:,:]

#uu = np.fft.fft(u,axis=0).real
#orth2 = uu[0,:,:].T@uu[0,:,:]