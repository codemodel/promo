import numpy as np
import csv

inputdir = 'c:/docker/promo/'
inputfile ='input.csv'
outputdir = 'c:/docker/promo/out/'
outputfile = 'output.csv'

#load static distribution
dist = np.loadtxt('distribution.csv', delimiter=',')


print('HELLO\t1.0')

for col in range(303):
    #read
    coldata = np.loadtxt(inputdir + inputfile, delimiter=',')

    #handle
    for i in range(1000):
        col_dist = dist[col]
        output = np.array([col_dist,]*1000)
        
    #write
    np.savetxt(outputdir+outputfile, output, delimiter=",")

    #update output filename for next round

    #wait for continue
    line = input()

    


