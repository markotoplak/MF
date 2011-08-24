
####
# EXAMPLE 1: 
####


# Import MF library entry point for factorization
import mf

# Construct sparse matrix in CSR format, which will be our input for factorization
from scipy.sparse import csr_matrix
from scipy import array
from numpy import dot
V = csr_matrix((array([1,2,3,4,5,6]), array([0,2,2,0,1,2]), array([0,2,3,6])), shape=(3,3))

# Print this tiny matrix in dense format
print V.todense()

# Run Standard NMF rank 4 algorithm
# Update equations and cost function are Standard NMF specific parameters (among others).
# If not specified the Euclidean update and Forbenius cost function would be used.
# We don't specify initialization method. Algorithm specific or random intialization will be used.
# In Standard NMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.
fit = mf.mf(V, method = "nmf", max_iter = 30, rank = 4, update = 'divergence', objective = 'div')

# Basis matrix. It is sparse, as input V was sparse as well.
W = fit.basis()
print "Basis matrix"
print W.todense()

# Mixture matrix. We print this tiny matrix in dense format.
H = fit.coef()
print "Coef"
print H.todense()

# Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler", fit.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print sparseness (Hoyer, 2004) of basis and mixture matrix
print "Sparseness Basis: %5.3f  Mixture: %5.3f" % (sm['sparseness'][0], sm['sparseness'][1])
# Print actual number of iterations performed
print "Iterations", sm['n_iter']

# Print estimate of target matrix V
print "Estimate"
print dot(W.todense(), H.todense())


####
# EXAMPLE 2: 
####


# Import MF library entry point for factorization
import mf

# Here we will work with numpy matrix
import numpy as np
V = np.matrix([[1,2,3],[4,5,6],[6,7,8]])

# Print this tiny matrix 
print V

# Run LSNMF rank 3 algorithm
# We don't specify any algorithm specific parameters. Defaults will be used.
# We don't specify initialization method. Algorithm specific or random intialization will be used. 
# In LSNMF case, by default random is used.
# Returned object is fitted factorization model. Through it user can access quality and performance measures.
# The fit's attribute `fit` contains all the attributes of the factorization.  
fit = mf.mf(V, method = "lsnmf", max_iter = 10, rank = 3)

# Basis matrix.
W = fit.basis()
print "Basis matrix"
print W

# Mixture matrix. 
H = fit.coef()
print "Coef"
print H

# Return the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
print "Distance Kullback-Leibler", fit.distance(metric = "kl")

# Compute generic set of measures to evaluate the quality of the factorization
sm = fit.summary()
# Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
print "Rss: %8.3f" % sm['rss']
# Print explained variance.
print "Evar: %8.3f" % sm['evar']
# Print actual number of iterations performed
print "Iterations", sm['n_iter']

# Print estimate of target matrix V 
print "Estimate"
print np.dot(W, H)