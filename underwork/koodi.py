
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib as pl
import matplotlib.pyplot as plt
import pylab
import pickle
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats import mode


# In[ ]:

#save_object(counts, 'distribution.pkl')
#dist = read_object('distribution.pkl')


# In[ ]:

def read_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
    
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[ ]:

data = pd.read_csv('c:/prob/proj/data1.txt',  sep=',', 
        encoding='utf-8-sig')

training = data[:50000]
testdata = data[50000:]
data = np.array(training)
testdata = np.array(testdata)


# In[ ]:

len(data[:,0])


# In[ ]:

def matchem(one, two):
  found = 0
  i=one
  j=two
  matching = np.zeros(shape=(len(data[:,0]),2))
  for line in range(len(data[:,0])): # each line
     if data[line,i]>0 and data[line,j] >0:
        matching[found]=[data[line,i], data[line,j]]
        found+=1
  #print('matchem, found: ', found)
  return matching[:found]


# In[ ]:

a = matchem(270, 291) #279, 291, 270


# In[ ]:

[data[1,270], data[1,291]]


# In[ ]:

a[1]


# In[ ]:

pearsonr(a[:,0], a[:,1])


# In[ ]:

np.count_nonzero(data)


# In[ ]:

#find nonzero count, mean and deviation for column
def getinfo(col):
  found=0  
  summa=0
  nonempty = np.zeros(len(data[:,0]))  
  for line in range(len(data[:,0])):
        if data[line,col]>0:
           found+=1
           nonempty[found] = data[line,col]
            
  if found >0:
     mycount = found
     mymean = np.mean(nonempty[:found])
     mystd = np.std(nonempty[:found])
     mymedian = np.median(nonempty[:found])
     mymode = mode(nonempty[:found])
     return [mycount, mymean, mystd, mymedian, mymode]
             
  else:
     return [0,0,0,0,0]


# In[ ]:

#get info of column
mycount, mymean, mystd, mymedian, mymode =  getinfo('270')

if mycount >1:
    print('count: ',mycount)
    print('mean: ',mymean)
    print('std: ', mystd)
    print('median: ', mymedian)
    print('mode: ', mymode)


# In[ ]:

a = getinfo('270')


# In[ ]:

#find nonzero count, mean and deviation for column
def get_colstats(data):
  col_stats = np.zeros(shape=(303,5))
  nonempty = np.zeros(len(data[:,0]))  
  for col in range(303):    
    found=0  
    summa=0
  
    for line in range(len(data[:,0])):
        if data[line,col]>0:
           found+=1
           nonempty[found] = data[line,col]
            
    if found >0:
     col_stats[col,0] = found
     col_stats[col,1] = np.mean(nonempty[:found])
     col_stats[col,2] = np.std(nonempty[:found])
     col_stats[col,3] = np.median(nonempty[:found])
     col_stats[col,4] = np.argmax(mode(nonempty[:found]))
        
  return col_stats


# In[ ]:

# get basic stats of columns, mean, std etc.

col_stats = get_colstats(data)
col_count= col_stats[:,0]
col_mean = col_stats[:,1]    
col_std  = col_stats[:,2]
col_median=col_stats[:,3]
col_mode = col_stats[:,4]
     


# In[ ]:

test = matchem(114,299)


# In[ ]:

len(test[:,0])


# In[ ]:

test = matchem(176,299)

dif =0
for i in range(len(test[:,0])):
  dif += abs(test[i,0] - test[i,1])
aver = dif/len(test)    
aver


# In[ ]:




# In[ ]:

len(data[np.nonzero(data[:,299])])



# In[ ]:

def countem(training):
       counts = np.zeros((303,100))
       #for line in range(len(training)):
       for line in range(len(training)):
          if line % 10000 ==0:
              print(line)
          for col in range(303):
                x = training[line, col]
                if x >0:
                 counts[col, x] +=1
       return counts


# In[ ]:

# Base line distributions for columns by coun value x from data and / n
def binsDist():
# make a distribution from table of counted numbers in data
# count total n of a column, add \little\ to n, each column is count(col)/n+little
# columns with 0 get little divided by them. if 41 might get, close to 0 smaller, how to distribute? maybe a minor thing
     
 dist = np.zeros((303,100))
      
 for col in range((len(counts))):
    total=0
    for value in range(100):
          total += counts[col][value]
          total +=0.01
    emptycols = 100- np.count_nonzero(counts[col][:])
    smoothing = 0.01 / emptycols
          #print(total)
    #for col in range((len(counts))):
    for value in range(100):
         if counts[col][value] > 0:
             dist[col][value] = counts[col][value] / total
         else: dist[col][value] = smoothing
 return dist


# In[ ]:

get_ipython().magic('time counts =countem(data)')


# In[ ]:

binsdist = binsDist()
binsdist


# In[ ]:

#Pearson correlations between column pairs
def correlation():
 #correlations
 #all_matches =np.zeros(shape=(303,303))    
 correlations = np.zeros(shape=(303,3))
 for i in range(303):
    correlations[i,2]=i
 matching = np.zeros(shape=(len(data[:,1]),2))

 #for i in range(290, len(data[1])):  #for column
 i=299

 #for j in range(len(data[1])): # compare to each other column,
 #for j in range((len(data[0]))): # compare to each other column    


 for j in range(len(data[0])): # compare to each other column            
     #j=1
     found=0
     for line in range(i, len(data[:,0])): # each value in column,
          if i!=j and data[line,i]>0 and data[line,j] >0:                #i!=j fixed to i<j  check if ok
               matching[found]=[data[line,i], data[line,j]]
               found+=1
     # limit - how many comparisons required              
     if found>100:
        # count pearson
        cut = matching[:found]
        
        
        #cor = pearsonr(cut[:,0], cut[:,1])
        cor = spearmanr(cut[:,0], cut[:,1])
        
        if (cor[0]>0.5 and cor[1]< 0.01):
          correlations[j,:2]=cor
          #correlations[j,2]=i

          if cor[0]==1: 
            print('cor 1 ',j)
          #print('i:',i,' j:',j)                 
          #print(cor)
          #print('found: ',found)
 #corsorted = nympy.sort(correlations)
            
 return correlations


# In[ ]:

get_ipython().magic('time correlations = correlation()')
best = np.argmax(correlations[:,0])
   
new = correlations[correlations[:,0].argsort()][::-1]
correlations = new


# In[ ]:

best


# In[ ]:

def estimateCol(correlations, inputdata, col):
 
 
 #best = 114   
 best = np.argmax(correlations[:,0])    
 
 # if no value in current line that matches correlating col, fallback 
 dist = binsdist[col]
 
 return dist
    
    
    


# In[ ]:

dist = estimateCol(correlations, inputdata, 299)
dist


# In[ ]:

inputdata = np.zeros(shape=(303, 2000))

#col = 298
totalscore =0
totalcount =0
# add range
for col in (299,299):
     testcol = testdata[:2000,col]
     inputdata[col]=testcol

     diff= 0
     summa=0
     count=0
     for i in range(len(testcol)):
        if testcol[i] >0:
            # for each row, find best distribution for column with that rows values
            dist = estimateCol(correlations, inputdata, col)
            #print(testcol[i],' est:', dist[testcol[i]] )
            summa +=dist[testcol[i]]
            count +=1
     if count >0:        
       #print('Average: ', summa/count)        
       totalscore += summa/count    
       totalcount +=1
     else:
       #print('No value at col ',col)
       1==1             
print('Average over sample: ', totalscore/ totalcount)


# In[ ]:

testcol = testdata[:2000,col]


# In[ ]:

np.nonzero(testdata[:2000,0])


# In[ ]:

test = matchem(114,299)


# In[ ]:

z = np.polyfit(matches[0], matches[1], 5) 


# In[ ]:

matches = matchem(114,299)


# In[ ]:

#Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
#Returns a vector of coefficients p that minimises the squared error.

#numpy.polyfit(x, y, deg, rcond=None, full=False)     #Least squares polynomial fit.
res = np.polyfit(matches[:,0], matches[:,1], 3, rcond=None, full=False)


# In[ ]:

res


# In[ ]:

def mypoly(x, res):
  result = res[0]*x**3 + res[1]*x**2 + res[2]*x**1 +res[3]
  print(result)
  return result


# In[ ]:

col_std[299]


# In[ ]:

mypoly(77, res)


# In[ ]:

matches = matchem(114,299)
for i in range(6):
  if matches[i,0] == 77:
    print(matches[i])


# In[ ]:

#try final distribution - polyfit
 bins = np.zeros(size=(100,100))  #col x, value 0-99  

    
 polyresult= np.zers(100)
 differs = np.zers(100)   
 #for i in range(len(data[0])): # compare to each other column,
  i =299      
   for j in range(i,(len(data[0]))): # compare to each other column    
      matches = matchem(i,j)
            
      #numpy.polyfit(x, y, deg, rcond=None, full=False)     #Least squares polynomial fit.
      res = np.polyfit(matches[:,0], matches[:,1], 3, rcond=None, full=False)
      diff=0           
      for line in range(len(matches)):             # see how i estimates j
         polyresult[line] = mypoly(matches[line,0], res)
         differs[line]= (polyresult[line] - matches[line,1]) 
  
                    


# In[ ]:

#try final distribution full bin count

from scipy.sparse import lil_matrix
#A = lil_matrix(303, 303)
A= np.empty([303, 303],dtype=dict)
#empty([2, 2], dtype=int)
#np.ones((100,), dtype=bool)
# 303 * 303/2 = 40.000 col pairs. for each 100x100 value pairs = 10.000. 40k * 10k = 40M
#    dict 

  
    
# estimator = np.zeros(size=303,303, 100, 100)
bins = np.zeros(shape=(100,100))  #col x, value 0-99  
toset = np.zeros(shape=(100,2)) #target yvalue, yprob

#for i in range(len(data[0])): # compare to each other column,
for i in range(0, 298):

   if i % 30 ==0:
     print(i)
   #for j in range(i,(len(data[0]))): # compare to each other column   
   j=299 #remove this
   if i<j:   
      matches = matchem(i,j)
      count=0 
      for line in range(len(matches)):             
          bins[matches[line,0], matches[line,1]] +=1      #bin 77,75 +=1 or 77,79+=1 etc.
      count=len(matches[:,0])      
     
      if count>20:  #handle counts in bin 75 -> {76,77}
                
         #level3 dict
         level3 = {}
         
         #for val in bins[np.nonzero(bins[:,0])]:
         for row in range(100):
             totalcount = np.sum(bins[row,:])
             #print(row)
             # create a struct for right side of 75 -> 75,76,77,78,89                        
             key = row
                    
             #create struct yvalue, prop(yvalue)        
             level4 = set([])
             myset = level4
                    
             for col in range(100):
                 if bins[row,col] !=0:
                   mytuple = (col, (bins[row,col]/totalcount))             
                   myset.add(mytuple)
                   mycopy=set.copy(myset)
                   level3[row]=mycopy
       
                            
         A[i,j] = level3    


# In[ ]:

input = testdata[0]
output = np.zeros(shape=(100,1))


# In[ ]:

j=299
for x in range(0,j):
    if input[x]>0:
        if not (A[i,j] is None):
           if not (A[i,j][test[col]] is None):
              print (A[i,j][test[col]])


# In[ ]:

#handle input line
for x in range (len(input)):
    #estimate distribution for x
    
    #check if earlier values exist, and their columns
    # order for best correlations for each col

 #fallback
 dist = binsdist[col]   


# In[ ]:

def estimateCol(correlations, inputdata, col):
 
 
 #best = 114   
 best = np.argmax(correlations[:,0])    
 
 # if no value in current line that matches correlating col, fallback 
 dist = binsdist[col]
 
 return dist
    

