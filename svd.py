import pandas as pd
import numpy as np
import csv
import math as mt
matrix={}
c=np.zeros((1000,1800))

def populate(fname):
    '''
        This function is used to make nested dictionary from the dataset.
        '''
    with open(fname) as f:
        for l in f:
            c[int(l.split()[0])][int(l.split()[1])]=float(l.split()[2])

#Converting dataset to nested dictionary.
populate("u1.base")

ratings_array = c
#print(ratings_array)
ratings_array = np.array(ratings_array)

def normalize(M):
    '''
        This function is used to normalize the matrix.
        Input- Matrix M.
        Output- Normalized Matrix M, avg.
    '''
    avg = 0
    M = np.array(M)
    count = 1
    for i in range(0,M.shape[0]):
        if ~(M[i] == 0):
            avg+=M[i]
            count+=1
    avg = avg/(count-1)
    for j in range (0,M.shape[0]):
        if ~(M[j]==0):
             M[j]-=avg  
    return M ,avg   

def SVD(ratings_array):
    '''
        Function to perform SVD decomposition of a matrix.
        Returns-
        U(2D numpy array): U matrix
        V_t(2D numpy array): Transpose of V matrix
        S(2D numpy array): Sigma Matrix
    '''
    ans = np.asmatrix(ratings_array)
#     print(ratings_array)
    answer = np.matmul(ans.transpose(),ans)

#     print(answer)
    eigenvalues, eigenvectors = np.linalg.eig(answer)
    V = eigenvectors
    V_t = V.transpose()
    eigenvalues = -np.sort(-eigenvalues)
    arr  =  eigenvalues
#     print(arr.size)
    # arr = arr[arr >= 0]
    arr[arr < 0] = 0
    eps = 1.915015330140815e-02
    arr[np.abs(arr) < eps] = 0
    size_t = arr.size
#     print(arr)
    arr = np.sqrt(arr)
#     print(arr)
    S = np.zeros((np.size(answer,0), np.size(answer,0)), float)
    row,col = np.diag_indices(arr.shape[0])

    S[row,col] = np.array(arr)

    S = S[~np.all(S == 0, axis=1)]
    S= np.delete(S,np.where(~S.any(axis=0))[0], axis=1)

    arr = arr[arr != 0]

    size_tt= arr.size
    pp = size_t - size_tt

    rrrrrrr = np.size(V_t,0)
    for i in range(0,pp):
          V_t= np.array(np.delete(V_t, rrrrrrr-pp,axis = 0))
  
    V = np.transpose(V_t)
    S_inverse = np.linalg.inv(np.matrix(S))
   
    U  = np.matmul(ratings_array,V_t.transpose())
    U = np.matmul(U,S_inverse)
    K =np.matmul(U,np.matmul(S,V_t))
    # K[np.abs(K) < eps] = 0
    return U,S,V_t


K=0

U,S,V_t = SVD(ratings_array)
#print("****   U   ******")
#print(U)
#print("****   S    ****")
#
#print(S)
#print("**** V transpose *****")
#print(V_t)
U,S,V_t = SVD(ratings_array)
K =np.matmul(U,np.matmul(S,V_t))
K=np.array(K)
print("**** Predicted Real matrix *****")
print(K)

def RMSE(M,T):
    '''
        This function is used to compute Root Mean Square Error between matrices M and T.
    '''
    X = 0
    cnt=0
    for i in range(0,M.shape[0]):
        for j in range(0,M.shape[1]):
            if(M[i][j]!=0):
                cnt+=1
                X = X+(M[i][j]-T[i][j])*(M[i][j]-T[i][j])
    return np.sqrt(X/cnt )

print("*********** RMSE 100%  *************")
print(RMSE(ratings_array,K))

def SE(M,P):
    '''
        This function is used to calculate spearman correlation.
    '''
    M = np.array(M)
    P = np.array(P)
    M =M-P
    val  = 0.
    for i in range(0,M.shape[0]):
        for j in range(0,M.shape[1]):
            val += M[i][j]*M[i][j]
    n =  M.shape[0]*M.shape[1]
    val =1 - val*6/(n*(n*n-1) )
    return val

print("*********** Spearman 100% *************")
print(SE(ratings_array,K))


def kpre(M,T):
    '''
        Calculating K-precision. To do this, we will assume that any true rating above 3 corresponds to a relevant item and any true rating below 3 is irrelevant.
    '''
    k_precision=0
    counttt=0
    counttt2=0
    for i in range(1,100):
        for j in range(1,100):
            if(M[i][j]>=3):
                counttt2+=1
                if(T[i][j]>=3):
                    counttt+=1


    return counttt/counttt2

print("*********** Precision on top K 100% *************")
print(kpre(ratings_array,K))

def ninety():
    '''
        This function is used for SVD with 90% retained energy.
    '''
    diagsum=0
    print(S)
    for i in range(0,S.shape[0]):
        for j in range(0,S.shape[1]):
            if(i==j):
                diagsum+=S[i][j]**2

    print(diagsum)

    limit=0.9
    num=0
    k=0
    flag=0
    print(S[0][0])
    for i in range(0,S.shape[0]):
        for j in range(0,S.shape[1]):
            if(i==j):
                num+=S[i][j]**2
                print(num)
                k+=1
                if(num/diagsum>=limit):
                    flag=1
                    break
            if(flag==1):
                break
    print(S.shape)
    print(k)
    sigma=S[:k,:k]
    Um=U[:,:k]
    Vm_t=V_t[:k,:]
    # print(sigma)
    print(sigma.shape)
    K =np.matmul(Um,np.matmul(sigma,Vm_t))
    print(K)
    rmse = RMSE(np.array(ratings_array),np.array(K))
    print("*********** RMSE 90% *************")
    print(rmse)
    print("*********** Spearman 90% *************")
    print(SE(np.array(ratings_array),np.array(K)))
    print("*********** Precision on top K 90% *************")
    print(kpre(np.array(ratings_array),np.array(K)))
ninety()
