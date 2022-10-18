import numpy as np
from numpy import linalg
import pywt

def tSVDdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    #cA = np.zeros((z2,n1,n2))
    arr = np.concatenate((cA, cD),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    cV1 = V1[0:z2, :, :]; cV2 = V1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    V1 = pywt.idwt(cV1,cV2, 'haar', axis=0)
    return U1, S1, V1

def teigdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n2))
    S1 = np.zeros((n0, n1, n2))

    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    arr = np.concatenate((cA, cD))
    for i in range(n0):
      M = arr[i, 0:]
      s, u = np.linalg.eig(M)
      idx = np.argsort(s)
      idx = idx[::-1][:n2]
      s = s[idx]
      u = u[:, idx]
      #s, u = linalg.cdf2rdf(S, U)
      np.fill_diagonal(S1[i,:,:],s.real)
      U1[i,:,:] = u.real
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    return S1,U1

def tproddwt(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    C = np.zeros((n0, n1, nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    #cA = np.zeros((z2, n1, n2))
    D = np.concatenate((cA, cD),axis=0)
    coeffsB = pywt.dwt(B, 'haar', axis=0)
    cAA, cDD = coeffsB
    #cAA = np.zeros((z2, nb, nc))
    Bhat = np.concatenate((cAA, cDD),axis=0)
    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])
    cC1 = C[0:z2, :, :]; cC2 = C[z2:z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return Cx
def tinvdwt(A):
    (n0, n1, n2) = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z2, :, :]
    cC2 = D2[z2:z1, :, :]

    D3 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return D3




def ttransx(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b))

    for j in range(a):
        B[j,:,:] = np.transpose(A[j,:,:])
    return B

def tSVDdwtdb_4(A):
    coeffs = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    arr = np.concatenate((coeffs[0], coeffs[1]), axis=0)
    (n0,n1,n2) =arr.shape
    U1 = np.zeros((n0, n1, n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0, n2, n2))
    for i in range(n0):
        M = arr[i, 0:]
        U, S, Vt = np.linalg.svd(M, full_matrices=True)
        np.fill_diagonal(S1[i, :, :], S)
        V1[i, :, :] = Vt.T
        U1[i, :, :] = U
    z2 = int(n0/2)
    cU1 = U1[0:z2, :, :];    cU2 = U1[z2:n0, :, :]
    cS1 = S1[0:z2, :, :];    cS2 = S1[z2:n0, :, :]
    cV1 = V1[0:z2, :, :];    cV2 = V1[z2:n0, :, :]
    U1 = pywt.idwt(cU1, cU2, 'db4', axis=0,mode='periodization')
    S1 = pywt.idwt(cS1, cS2, 'db4', axis=0,mode='periodization')
    V1 = pywt.idwt(cV1, cV2, 'db4', axis=0,mode='periodization')
    return U1, S1, V1


def tproddwtdb4(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    D = np.concatenate((coeffsA[0],coeffsA[1]), axis=0)
    coeffsB = pywt.dwt(B, 'db4', axis=0,mode='periodization')
    Bhat = np.concatenate((coeffsB[0], coeffsB[1]), axis=0)
    (z1,z2,z3) = Bhat.shape
    C = np.zeros((z1,n1,nc))
    for i in range(z1):
        C[i, :, :] = np.matmul(D[i, :, :], Bhat[i, :, :])
    cC1 = C[0:int(z1/2), :, :];    cC2 = C[int(z1/2):z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return Cx
def tinvdwt_db4(A):
    (n0, n1, n2) = A.shape
    z1 =int((n0+6)/2)
    coeffsA = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    D2 = np.zeros((z1+z1,n1,n2))
    for i in range(z1+z1):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z1, :, :]
    cC2 = D2[z1:z1+z1, :, :]

    D3 = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return D3

def fronorm(A):
    tmp = A*A
    B = np.absolute(tmp)
    C = np.sum(B)
    y = np.sqrt(C)
    return y
#########################kernel
def tproddwt4(A,B):
    (n0, n1, n2, n3) = A.shape
    (na, nb, nc, nd) = B.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    C = np.zeros((n0, n1, n2, nd))
    if n0 != na and n3 != nc:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD),axis=1)
    coeffsB = pywt.dwt(B, 'haar', axis=1)
    cAA, cDD = coeffsB
    Bhat = np.concatenate((cAA, cDD),axis=1)

    coeffsA = pywt.dwt(D, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD), axis=0)
    coeffsB = pywt.dwt(Bhat, 'haar', axis=0)
    cAA, cDD = coeffsB
    Bhat = np.concatenate((cAA, cDD), axis=0)
    for i in range(n0):
        for j in range(n1):
            C[i, j, :, :] = D[i, j, :, :] @ Bhat[i, j, :, :]
    cC1 = C[0:z2, :, :,:]; cC2 = C[z2:z1, :, :,:]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)

    cC1 = Cx[:, 0:z22, :,:];
    cC2 = Cx[:, z22:z11, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=1)
    return Cx


def ttransdwt4(A):
    n0, n1, n2, n3 = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    coeffs = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffs
    arr = np.concatenate((cA, cD), axis=1)
    coeffs2 = pywt.dwt(arr, 'haar', axis=0)
    cA2, cD2 = coeffs2
    arr2 = np.concatenate((cA2, cD2), axis=0)
    B = np.zeros((n0,n1,n3,n2))
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.transpose((arr2[i,j,:,:]))
    cC1 = B[0:z2, :, :, :];
    cC2 = B[z2:z1, :, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)

    cC1 = Cx[:, 0:z22, :, :];
    cC2 = Cx[:, z22:z11, :, :]
    B2 = pywt.idwt(cC1, cC2, 'haar', axis=1)
    return B2
def kernel2_G_dwt(A):
    d0,d1,d2,d3 = A.shape
    Gbar = np.zeros((d0,d1,d3,d3))
    a = np.zeros((d0,d1,d2,1))
    b = np.zeros((d0, d1,d2, 1))
    for k in range(d3):
          for j in range(d3):
            a[:,:,:,0] = A[:,:,:,j]
            b[:,:,:,0] = A[:,:,:,k]
            Gbar[:,:,j,k] = (np.abs((tproddwt4(ttransdwt4(a),b))+1)**0.8)[:,:,0,0]
    return Gbar

def kernel2_test_data_dwt(Train,Test):
    d0,d1,d2,d3 = Train.shape
    d00,d11,d22,d33 = Test.shape
    a = np.zeros((d0, d1,d2, 1))
    b = np.zeros((d0, d1,d2, 1))
    Gbar = np.zeros((d0,d2,d3,d33))
    for k in range(d33):
          for j in range(d3):
              a[:, :,:, 0] = Train[:, :,:, j]
              b[:, :,:, 0] = Test[:, :,:, k]
              Gbar[:,:,j,k] = (np.abs((tproddwt4(ttransdwt4(a),b))+1)**0.8)[:,:,0,0]
    return Gbar

######################################kernel
def Class_scatters2_dwt(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class))
    Sw =  np.zeros((n1,n2,n2))
    Sb = np.zeros((n1, n2, n2))
    aa = np.zeros((n1,n2,1))
    b = np.zeros((n1, n2, 1))
    Mean_tensor = np.zeros((n1, n2, 1))
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    Sa = np.zeros((n1, n2, n2))
    for i in range(num_class):


      occurrences = np.count_nonzero(y_train == i+1)
      Jn = np.ones((n1, int(occurrences), int(occurrences)))
      Jn = pywt.idwt(Jn[:int(n1/2), :, :], Jn[int(n1/2):int(n1), :, :], 'haar', axis=0)
      I = np.zeros((n1, int(occurrences), int(occurrences)))
      for ll in range(n1):
          I[ll,:,:] = np.eye(int(occurrences))
      II = pywt.idwt(I[:int(n1 / 2), :, :], I[int(n1 / 2):int(n1), :, :], 'haar', axis=0)
      Cn = II - Jn/int(occurrences)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      aa[:,:,0] = mean_tensor_train[:,:,i]
      H1 = tproddwt(Tensor_train[:,:,idx],Cn)
      H2 = tproddwt(H1,ttransx(Tensor_train[:,:,idx]))
      Sa = Sa + H2

    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddwt(b,ttransx(b)))*occurrences

    return Sa,Sb

def pred(U_tr0,U_tr1,U_tr2,U_tr3,U_tr4,U_tr5,U_tst0,U_tst1,U_tst2,U_tst3,U_tst4,U_tst5,test_labels, train_labels,num_class):
    (l,m,n) = U_tr0.shape
    (l1,m1,n1) = U_tst0.shape
    N0 = np.zeros((num_class,1))
    N1 = np.zeros((num_class, 1))
    N2 = np.zeros((num_class, 1))
    N3 = np.zeros((num_class, 1))
    N4 = np.zeros((num_class, 1))
    N5 = np.zeros((num_class, 1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    mean_tensor_train0 = np.zeros((l, m, num_class))
    mean_tensor_train1 = np.zeros((l, m, num_class))
    mean_tensor_train2 = np.zeros((l, m, num_class))
    mean_tensor_train3 = np.zeros((l, m, num_class))
    mean_tensor_train4 = np.zeros((l, m, num_class))
    mean_tensor_train5 = np.zeros((l, m, num_class))
    for i in range(num_class):
        occurrences = np.count_nonzero(train_labels == i)
        idx = np.where(train_labels == i)
        idx = idx[0]
        mean_tensor_train0[:, :, i] = (U_tr0[:, :, idx].sum(axis=2)) / occurrences
        mean_tensor_train1[:, :, i] = (U_tr1[:, :, idx].sum(axis=2)) / occurrences
        mean_tensor_train2[:, :, i] = (U_tr2[:, :, idx].sum(axis=2)) / occurrences
        mean_tensor_train3[:, :, i] = (U_tr3[:, :, idx].sum(axis=2)) / occurrences
        mean_tensor_train4[:, :, i] = (U_tr4[:, :, idx].sum(axis=2)) / occurrences
        mean_tensor_train5[:, :, i] = (U_tr5[:, :, idx].sum(axis=2)) / occurrences
    for i in range(n1):
        for j in range(num_class):
            d0 = np.linalg.norm(U_tst0[:, :, i] - mean_tensor_train0[:, :, j], ord='fro')
            d1 = np.linalg.norm(U_tst1[:, :, i] - mean_tensor_train1[:, :, j], ord='fro')
            d2 = np.linalg.norm(U_tst2[:, :, i] - mean_tensor_train2[:, :, j], ord='fro')
            d3 = np.linalg.norm(U_tst3[:, :, i] - mean_tensor_train3[:, :, j], ord='fro')
            d4 = np.linalg.norm(U_tst4[:, :, i] - mean_tensor_train4[:, :, j], ord='fro')
            d5 = np.linalg.norm(U_tst5[:, :, i] - mean_tensor_train5[:, :, j], ord='fro')
            N0[j, 0] = d0
            N1[j, 0] = d1
            N2[j, 0] = d2
            N3[j, 0] = d3
            N4[j, 0] = d4
            N5[j, 0] = d5
        idx0 = np.argmin(N0)
        vl0 = np.min(N0)
        idx1 = np.argmin(N1)
        vl1 = np.min(N1)
        idx2 = np.argmin(N2)
        vl2 = np.min(N2)
        idx3 = np.argmin(N3)
        vl3 = np.min(N3)
        idx4 = np.argmin(N4)
        vl4 = np.min(N4)
        idx5 = np.argmin(N5)
        vl5 = np.min(N5)
        IDX = [idx0,idx1,idx2,idx3,idx4,idx5]
        index = np.argmin([vl0,vl1,vl2,vl3,vl4,vl5])
        ClassTest[i, 0] = IDX[index]

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = ClassTest[i]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return ClassTest,test_pred, accuracy

def pred2(U_tr, U_tst, test_labels, train_labels):

    (l,m,n) = U_tr.shape

    (l1,m1,n1) = U_tst.shape

    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.linalg.norm(U_tst[:, :, i] - U_tr[:, :, j], ord='fro')
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy

def pred22(U_tr, U_tst, test_labels, train_labels):

    (m,n) = U_tr.shape

    (m1,n1) = U_tst.shape

    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.linalg.norm(U_tst[:, i] - U_tr[:, j])
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy

def Class_scatters_dwt(num_class,Tensor_train,y_train):
    n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n1,n2,num_class))
    Sw =  np.zeros((n1,n2,n2))
    Sb = np.zeros((n1, n2, n2))
    a = np.zeros((n1,n2,1))
    b = np.zeros((n1, n2, 1))
    Mean_tensor = np.zeros((n1, n2, 1))
    Mean_tensor[:,:,0] = (Tensor_train.sum(axis=2))/n3
    for i in range(num_class):
      Sa = np.zeros((n1, n2, n2))
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,i] = (Tensor_train[:,:,idx].sum(axis=2))/occurrences
      for j in idx:
          a[:,:,0] = Tensor_train[:,:,j]-mean_tensor_train[:,:,i]
          Sa = Sa + tproddwt(a,ttransx(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,0] = mean_tensor_train[:,:,i] - Mean_tensor[:,:,0]
        Sb = Sb + (tproddwt(b,ttransx(b)))*occurrences

    return Sw,Sb


def dwtdet(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    det = np.zeros((n0))
    for i in range(n0):
        det[i] = np.linalg.det(D[i,:,:])
    cC1 = det[0:z2]
    cC2 = det[z2:z1]
    D3 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return D3

def PREDICT(train,test):
    d0,d1,d2 = train.shape
    dis= np.zeros((d1))
    for i in range(d1):
        dis[i] = fronorm(train[:,:,i]-test[:,:,i])
    idx = np.argmin(dis)
    return idx

def quater(A):
    d0,d1,d2 = A.shape
    a0 = A[0,:,:]
    a1 = A[1,:,:]
    a2 = A[2,:,:]
    a3 = A[3,:,:]
    qtr = np.block([ [a0 + a1*1j, a2+a3*1j ],
                     [-1*a2 + a3*1j, a0-a1*1j] ])
    return qtr
def class_scatters_matrix(num_class,Tensor_train,y_train):
    n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n2,num_class))
    Sw =  np.zeros((n2,n2))
    Sb = np.zeros((n2, n2))
    a = np.zeros((n2,1))
    b = np.zeros((n2, 1))
    Mean_tensor = np.zeros((n2, 1))
    Mean_tensor[:,0] = (Tensor_train.sum(axis=1))/n3
    for i in range(num_class):
      Sa = np.zeros((n2, n2))
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,i] = (Tensor_train[:,idx].sum(axis=1))/occurrences
      for j in idx:
          a[:,0] = Tensor_train[:,j]-mean_tensor_train[:,i]
          Sa = Sa + (a@(a.T))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,0] = mean_tensor_train[:,i] - Mean_tensor[:,0]
        Sb = Sb + (b@(b.T))*occurrences

    return Sw,Sb
def pred_mtx(U_tr, U_tst, test_labels, train_labels):
    (m,n) = U_tr.shape
    (m1,n1) = U_tst.shape
    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.linalg.norm(U_tst[:, i] - U_tr[:, j])
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx
    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0
    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy
# A = np.random.rand(10,10,10)
# u,s,v = tSVDdwt(A)
# coeffs = pywt.dwt(A, 'haar', axis=0)
# Abar = np.concatenate((coeffs[0],coeffs[1]),axis=0)
# B = np.random.rand(10,10,10)
# B,s,v = tSVDdwt(B)
# coeffs = pywt.dwt(B, 'haar', axis=0)
# Bbar = np.concatenate((coeffs[0],coeffs[1]),axis=0)
# su = 0
# for i in range(10):
#     inner = (np.linalg.norm(Abar[i,:,:]))**2
#     su +=inner
# fro = su
# fro0 = fronorm(Abar)**2


# A = np.random.rand(10,10,10)
# AA = np.random.rand(10,10,10)
# u,s,v = tSVDdwt(A)
# u0,s0,v0 = tSVDdwt(AA)
# rot = np.array([[np.cos(30), -np.sin(30)],
#                 [np.sin(30),np.cos(30)]])
# B = np.zeros((10,2,2))
# for i in range(10):
#     B[i,:,:] = rot
# B = pywt.idwt(B[0:5,:,:],B[5:10,:,:],"haar",axis=0)
# Um0,Sm,Vm = tSVDdwt(np.random.rand(10,10,10))
# Um1,Sm,Vm = tSVDdwt(np.random.rand(10,10,10))
# U = tproddwt(u,Um0)
# U0 = tproddwt(u0,Um1)
# res1 = fronorm(tproddwt(u0,u))
# res2 = fronorm(tproddwt(U0,U))
# print(res1)
# print(res2)
