
"""
#######################################
Pmfcc (``methods.factorization.pmfcc``)
#######################################

**Penalized Matrix Factorization for Constrained Clustering (PMFCC)** [FWang2008]_. 

PMFCC is used for semi-supervised co-clustering. Intra-type information is represented as constraints to guide the factorization process. 
The constraints are of two types: (i) must-link: two data points belong to the same class, (ii) cannot-link: two data points cannot belong to the same class.

PMFCC solves the following problem. Given a target matrix V = [v_1, v_2, ..., v_n], it produces W = [f_1, f_2, ... f_rank], containing
cluster centers and matrix H of data point cluster membership values.    

Cost function includes centroid distortions and any associated constraint violations. Compared to the traditional NMF cost function, the only 
difference is the inclusion of the penalty term.  

.. literalinclude:: /code/methods_snippets.py
    :lines: 192-200
    
"""

from nimfa.models import *
from nimfa.utils import *
from nimfa.utils.linalg import *

import nimfa.methods.seeding.random

import operator

import time

def dotmore(*args):
    return reduce(dot, args)

def _separate_pn(a):
    return max(a, 0), min(a, 0)*(-1)

class Pmtfcc(smf.Smf):

    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param Theta: Constraint matrix (dimension: V.shape[1] x X.shape[1]). It contains known must-link (negative) and cannot-link 
                  (positive) constraints.
    :type Theta: `numpy.matrix`
    """

    def __init__(self, **params):
        self.name = "pmtfcc"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        smf.Smf.__init__(self, params)
        self.set_params()

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._Theta2_p, self._Theta2_n = _separate_pn(self.Theta2)
        self._Theta1_p, self._Theta1_n = _separate_pn(self.Theta1)

        for run in xrange(self.n_run):
            # [FWang2008]_; H = G2.T, W = S, G1=G1

            #just random seeding
            rs = nimfa.methods.seeding.random.Random()
            rs.max = self.V.max()
            rs.prng = np.random.RandomState()

            k = self.rank
            #rs.max = 1./self.V.shape[0]
            self.G1 = rs.gen_dense(self.V.shape[0], k)
            #rs.max = 1./self.V.shape[1]
            self.H = rs.gen_dense(k, self.V.shape[1]) 

            #for now k1 = k2 so we can use nimfa's seeding methods
            #self.G1, self.H = self.seed.initialize(
            #    self.V, self.rank, self.options)

            #H has to be non-negative

            #G1, H have to be non-negative
            self.H = max(self.H, 0)
            self.G1 = max(self.G1, 0)

            #compute S
            self.W = dotmore(inv_svd(dot(self.G1.T, self.G1)), 
                self.G1.T, self.V, self.H.T, inv_svd(dot(self.H, self.H.T)))

            p_obj = c_obj = sys.float_info.max
            best_obj = c_obj if run == 0 else best_obj
            iter = 0
            if self.callback_init:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback_init(mffit)
            while self.is_satisfied(p_obj, c_obj, iter):
                p_obj = c_obj if not self.test_conv or iter % self.test_conv == 0 else p_obj
                self.update()
                iter += 1
                c_obj = self.objective(
                ) if not self.test_conv or iter % self.test_conv == 0 else c_obj
                print iter, c_obj
                if self.track_error:
                    self.tracker.track_error(run, c_obj)
            if self.callback:
                self.final_obj = c_obj
                self.n_iter = iter
                mffit = mf_fit.Mf_fit(self)
                self.callback(mffit)
            if self.track_factor:
                self.tracker.track_factor(
                    run, W=self.W, H=self.H, final_obj=c_obj, n_iter=iter)
            # if multiple runs are performed, fitted factorization model with
            # the lowest objective function value is retained
            if c_obj <= best_obj or run == 0:
                best_obj = c_obj
                self.n_iter = iter
                self.final_obj = c_obj
                mffit = mf_fit.Mf_fit(copy.deepcopy(self))

        mffit.fit.tracker = self.tracker
        return mffit

    def is_satisfied(self, p_obj, c_obj, iter):
        """
        Compute the satisfiability of the stopping criteria based on stopping parameters and objective function value.
        
        Return logical value denoting factorization continuation. 
        
        :param p_obj: Objective function value from previous iteration. 
        :type p_obj: `float`
        :param c_obj: Current objective function value.
        :type c_obj: `float`
        :param iter: Current iteration number. 
        :type iter: `int`
        """
        if self.max_iter and self.max_iter <= iter:
            return False
        if self.test_conv and iter % self.test_conv != 0:
            return True
        if self.min_residuals and iter > 0 and p_obj - c_obj < self.min_residuals:
            return False
        if iter > 0 and c_obj > p_obj:
            return False
        return True

    def set_params(self):
        """Set algorithm specific model options."""
        self.Theta2 = self.options.get('Theta2', sp.csr_matrix((self.V.shape[1], self.V.shape[1])))
        self.Theta1 = self.options.get('Theta1', sp.csr_matrix((self.V.shape[0], self.V.shape[0])))
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track(
        ) if self.track_factor and self.n_run > 1 or self.track_error else None

    def update(self):
        """Update basis and mixture matrix."""
        # [FWang2008]_; H = G2.T, W = S, G1=G1

        G2 = self.H.T
        G1 = self.G1
        S = self.W
        R = self.V
        T1n, T1p = self._Theta1_n, self._Theta1_p
        T2n, T2p = self._Theta2_n, self._Theta2_p

        ts = time.time()
        dotG1 = dot(G1.T, G1)
        dotG2 = dot(G2.T, G2)

        if np.max(dotG1) > 1e100 or np.max(dotG2) > 1e100 : #it can loop in inv_svd
            raise np.linalg.linalg.LinAlgError()

        def addeps(denom):
            return denom + np.finfo(float).eps

        S = dotmore(inv_svd(dotG1), G1.T, R, G2, inv_svd(dotG2))

        p1_p, p1_n = _separate_pn(dotmore(R, G2, S.T))
        p22_p, p22_n = _separate_pn(dotmore(S, G2.T, G2, S.T)) #error in the article (swapped S and S.T)
        enum = p1_p + dot(G1, p22_n) + T1n.dot(G1) #direct calls as they avoid intermediate contructions
        denom = p1_n + dot(G1, p22_p) + T1p.dot(G1)
        denom = addeps(denom)
        #G1 = multiply(G1, sop(elop(enum, denom, div), s=None, op=np.sqrt))
        G1 = np.multiply(G1, np.sqrt(enum/denom)) #five times faster
        
        p1_p, p1_n = _separate_pn(dotmore(R.T, G1, S))
        p22_p, p22_n = _separate_pn(dotmore(S.T, G1.T, G1, S)) #error in the article (swapped S and S.T)
        enum = p1_p + dot(G2, p22_n) + T2n.dot(G2) #direct calls avoid intermediate constructions of dot
        denom = p1_n + dot(G2, p22_p) + T2p.dot(G2)
        denom = addeps(denom)
        #G2 = multiply(G2, sop(elop(enum, denom, div), s=None, op=np.sqrt))
        G2 = np.multiply(G2, np.sqrt(enum/denom))

        print "all", time.time() - ts

        self.W = S
        self.H = G2.T
        self.G1 = G1

    def objective(self):
        """Compute Frobenius distance cost function with penalization term."""
        #n = power(self.V - dotmore(self.G1, self.W, self.H), 2).sum()
        n = np.linalg.norm(self.V - dotmore(self.G1, self.W, self.H))**2
        r = trace(dot(self.H, self.Theta2.dot(self.H.T))) + trace(dot(self.G1.T, self.Theta1.dot(self.G1)))
        return n+r

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
