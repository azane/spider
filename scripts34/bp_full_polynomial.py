"""The test round with bayespy for curve fitting with high order polynomials.

Some data is generated and prepared in <PREP>.

A bayespy model is constructed in <MODEL>

New data is generated and tested against the polynomial in <EXAMINE>
"""

import bayespy as bp
import numpy as np
import matplotlib.pyplot as plt


def gen_indData(size=500):
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
    """Multiplies and exponentiates terms in preperation for beta application. This is the full value of the term without the coefficient."""
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

#generate some data.
indData = gen_indData(size=500)
depData = gen_depData(indData)

n = 3 #set the max degree of the polynomial.
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

#print results from data actually used
absDiff = np.absolute(depData - np.einsum('ij,...ij', betas.random(), sansBetas))
print ("mean dif actual %: " + str(np.absolute((absDiff/depData).mean())*100))


#print results from newly generated data
for i in range(20):
    indData = gen_indData(size=30000)
    depData = gen_depData(indData)
    sansBetas = pre_beta(indData, subsetIndices, n)
    
    absDiff = np.absolute(depData - np.einsum('ij,...ij', betas.random(), sansBetas))
    print ("mean dif regen %: " + str(np.absolute((absDiff/depData).mean())*100))


plt.show(block=True)

#----</EXAMINE>----