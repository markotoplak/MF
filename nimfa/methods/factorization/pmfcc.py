
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

import operator

import time

def _separate_pn(a):
    return max(a, 0), min(a, 0)*(-1)

class Pmfcc(smf.Smf):

    """
    For detailed explanation of the general model parameters see :mod:`mf_run`.
    
    The following are algorithm specific model options which can be passed with values as keyword arguments.
    
    :param Theta: Constraint matrix (dimension: V.shape[1] x X.shape[1]). It contains known must-link (negative) and cannot-link 
                  (positive) constraints.
    :type Theta: `numpy.matrix`
    """

    def __init__(self, **params):
        self.name = "pmfcc"
        self.aseeds = ["random", "fixed", "nndsvd", "random_c", "random_vcol"]
        smf.Smf.__init__(self, params)
        self.set_params()

    def factorize(self):
        """
        Compute matrix factorization.
         
        Return fitted factorization model.
        """
        self._Theta_p, self._Theta_n = _separate_pn(self.Theta)

        for run in xrange(self.n_run):
            # [FWang2008]_; H = G.T, W = F (Table 2)
            self.W, self.H = self.seed.initialize(
                self.V, self.rank, self.options)
            #H has to be non-negative
            self.H = max(self.H, 0)
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
                #print iter, c_obj
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
        self.Theta = self.options.get('Theta', sp.csr_matrix((self.V.shape[1], self.V.shape[1])))
        self.track_factor = self.options.get('track_factor', False)
        self.track_error = self.options.get('track_error', False)
        self.tracker = mf_track.Mf_track(
        ) if self.track_factor and self.n_run > 1 or self.track_error else None

    def update(self):
        """Update basis and mixture matrix."""

        dotH = dot(self.H, self.H.T)
        print "max dotH", np.max(dotH)
        if np.max(dotH) > 1e10: #it can loop in inv_svd, it is problematic optimization anyway
            raise np.linalg.linalg.LinAlgError()
        self.W = dot(self.V, dot(self.H.T, inv_svd(dotH)))

        #assume W and H are not sparse and that theta
        #is either sparse or in CSR

        FtF = dot(self.W.T, self.W)
        XtF = dot(self.V.T, self.W)
        FtF_p, FtF_n = _separate_pn(FtF)
        XtF_p, XtF_n = _separate_pn(XtF)

        Theta_n_G = self._Theta_n.dot(self.H.T)
        Theta_p_G = self._Theta_p.dot(self.H.T)

        GFtF_p = dot(self.H.T, FtF_p)
        GFtF_n = dot(self.H.T, FtF_n)
        
        enum = XtF_p + GFtF_n + Theta_n_G
        denom = XtF_n + GFtF_p + Theta_p_G

        denom = denom.todense() + np.finfo(float).eps if sp.isspmatrix(
            denom) else denom + np.finfo(float).eps
        Ht = multiply(
            self.H.T, sop(elop(enum, denom, div), s=None, op=np.sqrt))

        self.H = Ht.T

    def objective(self):
        """Compute Frobenius distance cost function with penalization term."""
        n =  np.linalg.norm(self.V - dot(self.W, self.H))**2
        o =  trace(dot(self.H, self.Theta.dot(self.H.T)))
        return n+o

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
