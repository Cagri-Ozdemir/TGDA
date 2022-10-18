import numpy as np
from wavelet import tSVDdwt,tproddwt,ttransx,Class_scatters2_dwt,tinvdwt,teigdwt,pred22,pred2
import pywt
import scipy.io
from fft_fft import fronorm
import pickle
df1 = scipy.io.loadmat('bend.mat');df2 = scipy.io.loadmat('jack.mat');df3 = scipy.io.loadmat('jump.mat');
df4 = scipy.io.loadmat('pjump.mat');df5 = scipy.io.loadmat('run.mat');df6 = scipy.io.loadmat('side.mat');
df7 = scipy.io.loadmat('skip.mat');df8 = scipy.io.loadmat('walk.mat');df9 = scipy.io.loadmat('wave1.mat');df10 = scipy.io.loadmat('wave2.mat');
df1 =df1["bend"];df2 =df2["jack"];df3 =df3["jump"];df4 =df4["pjump"];df5 =df5["run"];df6 =df6["side"];df7 =df7["skip"];
df8 =df8["walk"];df9 =df9["wave1"];df10 =df10["wave2"];
data_frame = np.concatenate((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10),axis=3)
sub1=np.zeros((20,20,20,10)); sub2=np.zeros((20,20,20,10)); sub3=np.zeros((20,20,20,10)); sub4=np.zeros((20,20,20,10))
sub5=np.zeros((20,20,20,10)); sub6=np.zeros((20,20,20,10)); sub7=np.zeros((20,20,20,10)); sub8=np.zeros((20,20,20,10))
sub9=np.zeros((20,20,20,10))
k = 0
for i in range((10)):
    sub1[:,:,:,i] = data_frame[:,:,:,k]
    sub2[:, :, :, i] = data_frame[:,:,:,k+1]
    sub3[:, :, :, i] = data_frame[:,:,:,k+2]
    sub4[:, :, :, i] = data_frame[:,:,:,k+3]
    sub5[:, :, :, i] = data_frame[:,:,:,k+4]
    sub6[:, :, :, i] = data_frame[:,:,:,k+5]
    sub7[:, :, :, i] = data_frame[:,:,:,k+6]
    sub8[:, :, :, i] = data_frame[:,:,:,k+7]
    sub9[:, :, :, i] = data_frame[:,:,:,k+8]
    k+=9
# Train_set = np.concatenate((sub2,sub3,sub4,sub5,sub6,sub7,sub8,sub9),axis=3)
Train_set = np.concatenate((sub2,sub3,sub4,sub5,sub6,sub7,sub8,sub9),axis=3)#################select TRAIN subjects!!!!!!!!!!!
Test_set = sub1###########################################################select TEST subject!!!!!!!!!!!!!!!!!
# Test_set = np.concatenate((sub7,sub7,sub7,sub7,sub7,sub7,sub7,sub7),axis=3)

######create labels of training & testing sets#####
k = 0
Y_test = np.zeros(10)
for i in range(10):
    for ii in range(1):
        Y_test[k] = i+1
        k+= 1
k = 0
Y_train = np.concatenate((Y_test,Y_test,Y_test,Y_test,Y_test,Y_test,Y_test,Y_test))
########################
########################
train_subspaces_U1 = {}
train_subspaces_U2 = {}
train_subspaces_U3 = {}
train_subspaces_V1 = {}
train_subspaces_V2 = {}
train_subspaces_V3 = {}
eig = 4
for i in range(len(Y_train)):
    ut, st, vt = tSVDdwt(Train_set[:,:,:,i])
    ut2, st2, vt2 = tSVDdwt(np.swapaxes(Train_set[:,:,:,i], 0, 1))
    ut3, st3, vt3 = tSVDdwt(np.swapaxes(Train_set[:,:,:,i], 0, 2))
    train_subspaces_U1[i] = ut[:, :, :eig]
    train_subspaces_U2[i] = ut2[:, :, :eig]
    train_subspaces_U3[i] = ut3[:, :, :eig]
    train_subspaces_V1[i] = vt[:, :, :eig]
    train_subspaces_V2[i] = vt2[:, :, :eig]
    train_subspaces_V3[i] = vt3[:, :, :eig]
##################################################################################################################
test_subspaces_U1 = {}
test_subspaces_U2 = {}
test_subspaces_U3 = {}
test_subspaces_V1 = {}
test_subspaces_V2 = {}
test_subspaces_V3 = {}
for i in range(len(Y_test)):
    ute, ste, vte = tSVDdwt(Test_set[:,:,:,i])
    ute2, ste2, vte2 = tSVDdwt(np.swapaxes(Test_set[:,:,:,i], 0, 1))
    ute3, ste3, vte3 = tSVDdwt(np.swapaxes(Test_set[:,:,:,i], 0, 2))
    test_subspaces_U1[i] = ute[:, :, :eig]
    test_subspaces_U2[i] = ute2[:, :, :eig]
    test_subspaces_U3[i] = ute3[:, :, :eig]
    test_subspaces_V1[i] = vte[:, :, :eig]
    test_subspaces_V2[i] = vte2[:, :, :eig]
    test_subspaces_V3[i] = vte3[:, :, :eig]
################################################################################################################
grass_ker_u1 = np.zeros((6,len(Y_train),len(Y_train)))
for jj in range(len(Y_train)):
  for ii in range(len(Y_train)):
   grass_ker_u1[0, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_U1[jj]), train_subspaces_U1[ii]))**2
   grass_ker_u1[1, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_U2[jj]), train_subspaces_U2[ii]))**2
   grass_ker_u1[2, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_U3[jj]), train_subspaces_U3[ii]))**2
   grass_ker_u1[3, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V1[jj]), train_subspaces_V1[ii]))**2
   grass_ker_u1[4, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V2[jj]), train_subspaces_V2[ii]))**2
   grass_ker_u1[5, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V3[jj]), train_subspaces_V3[ii]))**2

test_grass_ker_u1 = np.zeros((6,len(Y_train),len(Y_test)))
for jj in range(len(Y_train)):
  for ii in range(len(Y_test)):
   test_grass_ker_u1[0,jj,ii] = fronorm(tproddwt(ttransx(train_subspaces_U1[jj]), test_subspaces_U1[ii]))**2
   test_grass_ker_u1[1, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_U2[jj]), test_subspaces_U2[ii]))**2
   test_grass_ker_u1[2, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_U3[jj]), test_subspaces_U3[ii]))**2
   test_grass_ker_u1[3, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V1[jj]), test_subspaces_V1[ii]))**2
   test_grass_ker_u1[4, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V2[jj]), test_subspaces_V2[ii]))**2
   test_grass_ker_u1[5, jj, ii] = fronorm(tproddwt(ttransx(train_subspaces_V3[jj]), test_subspaces_V3[ii]))**2

Sw,Sb = Class_scatters2_dwt(10, grass_ker_u1, Y_train)
II = np.eye(len(Y_train),len(Y_train))
III = np.zeros((6,len(Y_train),len(Y_train)))
for i in range(6):
  III[i,:,:] = II*1.e-3
III = pywt.idwt(III[:3,:,:],III[3:6,:,:], 'haar', axis=0)
Sww = Sw + III
S = tproddwt(tinvdwt(Sww), Sb)
SS, UU = teigdwt(S)
u = UU[:, :, :9]
pro_df_trn0 = tproddwt(ttransx(u), grass_ker_u1)
pro_df_tst0 = tproddwt(ttransx(u), test_grass_ker_u1)
test_pred, accuracy = pred2(pro_df_trn0,pro_df_tst0,Y_test,Y_train)
print("accuracy:",accuracy)