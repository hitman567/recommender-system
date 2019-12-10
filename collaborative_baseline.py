import math
import collections
import operator
import pandas as pd
import numpy as np
from numpy import *
from random import uniform
matrix={}
c=np.zeros((1000,1800))

for i in range(1000):
    matrix[i]={}

def populate(fname):
    '''
        This function is used to make nested dictionary from the dataset.
    '''
    with open(fname) as f:
        for l in f:
            matrix[int(l.split()[0])][int(l.split()[1])]=float(l.split()[2])
            c[int(l.split()[0])][int(l.split()[1])]=float(l.split()[2])

#Converting dataset to nested dictionary.
populate("u1.base")

print(c)
df=pd.DataFrame(c)
df.to_csv("test_matrix.csv")
def calcmean():
    '''
        This function is used to calculate overall mean of the matrix.
    '''
    means=0
    counttot=0
    for key in matrix:
        for movie in matrix[key].keys():
            means+=matrix[key].get(movie)
            counttot+=1
    return means/counttot

total_mean=calcmean()
baseline_users = []
non_zero = 0
for i in range(1, c.shape[0]):
    base = 0.0
    non_zero = 0
    for j in range(1, c.shape[1]):
        if  c[i][j] != 0 :
            print(i,j)
            base = base + c[i][j]
            non_zero = non_zero + 1
    if(non_zero!=0):
        base = base / non_zero
        base = base - total_mean
        baseline_users.append(base)
    else:
        baseline_users.append(0)

baseline_movies = []
non_zero = 0
for i in range(1, c.shape[1]):
    base = 0.0
    non_zero = 0
    for j in range(1, c.shape[0]):
        if c[j][i] != 0:
            base = base + c[j][i]
            non_zero = non_zero + 1
    if(non_zero!=0):
        base = base / non_zero
        base = base - total_mean
        baseline_movies.append(base)
    else:
        baseline_movies.append(0)

print(baseline_users)
print(baseline_movies)
print("YES")
# In[6]:

print(baseline_users[1])
print(baseline_movies[1])
bxi = np.zeros(c.shape)
for i in range(1,800):
    for j in range(1,1000):
        #print(i,j)
        #if(baseline_movies[j]!=0)
        bxi[i][j] = total_mean + baseline_users[i] + baseline_movies[j]

#def basevalue(user,i,mean):
#    buser=0
#    bmovie=0
#    count1=0
#    count2=0
#    for movie in matrix[user].keys():
#        count1=count1+1
#        buser=buser+(matrix[user].get(movie))
#
#    if(count1!=0):
#        buser=buser/count1
#        buser=buser-mean
#    else:
#        buser=0
#
#    for key in matrix:
#        itemi=matrix[key].get(i)
#        if itemi is not None:
#            count2=count2+1
#            bmovie=bmovie+(itemi)
#    if(count2!=0):
#        bmovie=bmovie/count2
#        bmovie=bmovie-mean
#    else:
#        bmovie=0
#
#    return (mean+bmovie+buser)


def pearsoncoff(x,y):
    '''
        This function calculates pearson coefficient between two items x and y.
        Returns a float pearson coefficient.
    '''
    xmean=0
    ymean=0
    countx=0
    county=0
    for key in matrix:
        itemx=matrix[key].get(x)
        itemy=matrix[key].get(y)
        if itemx is not None:
            xmean=xmean+itemx
            countx=countx+1
        if itemy is not None:
            ymean=ymean+itemy
            county=county+1
    
    if(countx==0):
        xmean=0
    else:
        xmean=xmean/countx
    
    if(county==0):
        ymean=0
    else:
        ymean=ymean/county
    
    topxy=0
    botx=0
    boty=0
    for key in matrix:
        itemx=matrix[key].get(x)
        itemy=matrix[key].get(y)
        if itemx is not None and itemy is not None:
            topxy=topxy+(itemx-xmean)*(itemy-ymean)
            botx=botx+(itemx-xmean)**2
            boty=boty+(itemy-ymean)**2

    botx=math.sqrt(botx)
    boty=math.sqrt(boty)

    botxy=botx*boty
    if(botxy==0):
        return 0
    else:
        return topxy/botxy

def rating(x,i,total_mean):
    '''
        This function predicts the rating which user x would give to movie i. This function implments item-item filtering.
        Returns a float predicted rating.
    '''
    topsim={}
    num=0
    den=0
    
    for movie in matrix[x].keys():
        topsim[movie]=pearsoncoff(movie,i)
    
    sorted_top_items=sorted(topsim.items(),key=operator.itemgetter(1),reverse=True)
    
    count=0
    for movie in sorted_top_items[:20]:
        if(movie[1]>0):
            num+=abs(movie[1]*(matrix[x].get(movie[0])))
            den=den+movie[1]
        
        count+=1
    num+=bxi[x][i]

    if(den==0):
        return 3
    return min(uniform(4.65,4.85),num/den)



def recommend(user,total_mean):
    '''
        This function returns a list of top 10 movie recommendations for user.
    '''
    recomlist={}
    
    for i in range(1,1680):
        if matrix.get(user).get(i) is None:
            print(i)
            recomlist[i]=rating(user,i,total_mean)
            print(recomlist[i])
    
    sorted_recom_list=sorted(recomlist.items(),key=operator.itemgetter(1),reverse=True)
    
    for i in range(1,11):
        print(sorted_recom_list[i])

#total_mean=calcmean()
print(total_mean)

a=pd.DataFrame(matrix)

where_are_nans=isnan(a)
a[where_are_nans]=0
n=a.shape[0]
m=a.shape[1]
b=np.zeros((1000,1800))

#recommend(1,total_mean)

for key in range(1,100):
    for movie in range(1,100):
        rate=matrix[key].get(movie)
        if rate is not None:
            b[key][movie]=rating(key,movie,total_mean)
            print(matrix[key].get(movie),b[key][movie])


for key in range(1,100):
    for movie in range(1,100):
        rate=c[key][movie]
        if rate!=0 :
            b[key][movie]=rating(key,movie,total_mean)
            print(c[key][movie],b[key][movie])


for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            b[i][j]=rating(i,j,total_mean)
            print(a[i][j],b[i][j])

for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            print(a[i][j],b[i][j])


rmse=0
count = 0
for i in range(1,100):
    for j in range(1,100):
        rate=c[i][j]
        if rate!=0 :
            rmse += (b[i][j]-c[i][j])**2
            count = count + 1
rmse=math.sqrt(rmse/count)

print(rmse)

# Calculating spearman correlation
spearman=0
dterm=0
cnt=0
for i in range(1,100):
    for j in range(1,100):
        if(c[i][j]!=0):
            dterm+=(c[i][j]-b[i][j])**2
            cnt+=1
spearman=1-6*dterm/(cnt*(cnt*cnt-1))
print(spearman)

'''
    Calculating K-precision. To do this, we will assume that any true rating above 3 corresponds to a relevant item and any true rating below 3 is irrelevant.
'''
k_precision=0
counttt=0
counttt2=0
for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]>=3):
            counttt2+=1
            if(b[i][j]>=3):
                counttt+=1


k_precision=counttt/counttt2
print(k_precision)


