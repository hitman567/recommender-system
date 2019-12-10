import math
import collections
import operator
import pandas as pd
import numpy as np
from numpy import *
matrix={}

for i in range(1000):
    matrix[i]={}

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

def rating(x,i):
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
        num+=movie[1]*(matrix[x][movie[0]])
        if(movie[1]<0):
            den=den-movie[1]
        else:
            den=den+movie[1]
        count+=1
    
    if(den==0):
        return 3
    return num/den

def populate(fname):
    '''
        This function is used to make nested dictionary from the dataset.
    '''
    with open(fname) as f:
        for l in f:
            matrix[int(l.split()[0])][int(l.split()[1])]=float(l.split()[2])

            
            
def recommend(user):
    '''
        This function returns a list of top 10 movie recommendations for user.
    '''
    recomlist={}
    
    for i in range(1,1683):
        if matrix.get(user).get(i) is None:
            recomlist[i]=rating(user,i)
            print (i)
    
    sorted_recom_list=sorted(recomlist.items(),key=operator.itemgetter(1),reverse=True)
    
    for i in range(1,11):
        print(sorted_recom_list[i])
        

#Converting dataset to nested dictionary.
populate("u1.base")

a=pd.DataFrame(matrix)
#print(a)
where_are_nans=isnan(a)
a[where_are_nans]=0
n=a.shape[0]
m=a.shape[1]
b=np.zeros((n,m))

#recommend(1)

for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            b[i][j]=rating(i,j)
            print(i,j)

for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            print(a[i][j],b[i][j])

rmse=0
count = 0
for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            rmse+= (b[i][j]-a[i][j])**2
            count = count +1
rmse=math.sqrt(rmse/count)
print(rmse)

# Calculating spearman correlation
spearman=0
dterm=0
cnt=0
for i in range(1,100):
    for j in range(1,100):
        if(a[i][j]!=0):
            dterm+=(a[i][j]-b[i][j])**2
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
