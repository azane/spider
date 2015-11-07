import bayespy as bp
import numpy as np
import matplotlib.pyplot as plt

def gen_indData(size=5000):
    #size = size
    d1 = np.random.uniform(low=30, high=70, size=size)
    d2 = np.random.normal(0, 5, size=size)
    d3 = np.random.uniform(-50, 0, size=size)
    #d4 = np.random.uniform(low=5, high=6, size=size)
    
    dat = np.vstack((d1,d2,d3))
    
    return dat
    
def gen_depData(indData):
    
    #not clean, but easy to visualize below.
    a = indData[0]
    b = indData[1]
    c = indData[2]
    
    dat = 5*(a*b)**3 - 0.5*(c)**2 + 2*b
    
    #noise it up.
    #dat = dat * (np.random.normal(loc=0.0, scale=0.01, size=(dat.size)) + 1)
    
    return dat

def dim_indices(indData):
    """Returns a list of unequal numpy arrays holding indices corresponding to indData"""
    x = indData
    
    d = np.ma.size(x, 0) #infer dimensions from independent data
    c = 2**d #calculate number of dimension subsets.
    
    indexList = []
    
    for i in range(c):
        bs = np.array([(np.binary_repr(i, width=np.ma.size(x, 0)))])[0] #get boolean string (bs) from decimal
        ba = np.array(list(bs)) #get boolean array (ba) from boolean string
        dimIndices = np.flatnonzero(ba.astype(int)) #get the indices of the dimensions to be included in this term.
        
        #print (str(i) + " == " + bs + " == " + str(dimIndices)) #display decimal -> boolean string -> dimension indices
        indexList.append(dimIndices)
    
    return indexList

def pre_beta(indData, indexList, degree):
    """Multiplies and exponentiates terms in preperation for beta application."""
    x = indData
    n = degree
    
    
    l = []
    for i in indexList:
        l.append(indData[i].prod(axis=0))
        
    sans_exponentiation = np.array(l)
    
    l = []
    for i in range(n):
        l.append(sans_exponentiation**(i+1))#append the exponentiated list of terms.
    
    sans_beta = np.array(l)
    sans_beta = np.einsum('ijk->kij', sans_beta)#roll axes so s isn't summed over? this makes it work...don't know why. #FIXME
    
    return sans_beta
#----<PREP>----#

data = np.genfromtxt('data/full_with_n.csv', delimiter=',') #pull in data from csv.

#the [:,i] effectively switches rows to columns...maybe just switching rows and columns would be easier? ; )
indData = np.vstack((data[:,0], data[:,1], data[:,-2])) #get the first, second, and second to last columns as independent data.
depData = data[:,-1] #get the last row, the dependent data.

n = 6 #set the max degree of the polynomial.
subsetIndices = dim_indices(indData)

sansBetas = pre_beta(indData, subsetIndices, n)

#----</PREP>----#

#----<MODEL>----
#
betas = bp.nodes.GaussianARD(0, 0.001, shape=(n,2**np.ma.size(indData, 0))) #a beta for every term. each variable has beta for each degree.

#print(np.einsum('ij,...ij', betas.random(), expd_terms))

f = bp.nodes.SumMultiply('ij,ij', betas, sansBetas) #the dot product over the betas and expd_terms

tau = bp.nodes.Gamma(1e-3, 1e-3) #noise parameter.

y = bp.nodes.GaussianARD(f, tau)#, plates=(np.ma.size(depData),))
#----</MODEL>----

#----<INFERENCE>---

y.observe(depData)

Q = bp.inference.VB(betas, tau, y)

Q.update(repeat=10)
#----</INFERENCE>----

#----<EXAMINE>-----
#generate some artificial independent data by proportionally noising up the fitted data.
s = np.ma.size(indData, 1)
muscleLength = indData[0] * (np.random.normal(0, 0.1, size=s) + 1)
balance = indData[1] * (np.random.normal(0, 0.1, size=s) + 1)
tBack = indData[2] #tBack will never be noised.

#gather indData
test_indData = np.vstack((muscleLength, balance, tBack))

#generate sansBetas
test_sansBetas = pre_beta(test_indData, subsetIndices, n)
#test_sansBetas = pre_beta(indData, subsetIndices, n) #use fitted data.

#generate dependent data from the fitted poly.
fromFitted = np.einsum('ij,...ij', betas.random(), test_sansBetas)

#gather arrays to send to the csv, swap rows and columns.
toCSV = np.vstack((muscleLength, balance, tBack, fromFitted)).transpose()
#toCSV = np.vstack((indData[0], indData[1], indData[2], fromFitted)).transpose() #use fitted data.

#store it in a csv to be plotted.
np.savetxt("data/VB_fit.csv", toCSV, fmt='%.3f', delimiter=",")


plt.show(block=True)

#----</EXAMINE>----