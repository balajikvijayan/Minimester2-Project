""" Non-negative matrix factorization
"""
# Author: Vlad Niculae
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# Author: Chih-Jen Lin, National Taiwan University (original projected gradient
#     NMF implementation)
# Author: Anthony Di Franco (original Python and NumPy port)
# License: BSD 3 clause
from __future__ import division

from math import sqrt
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_random_state, check_array
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import check_is_fitted


def safe_vstack(Xs):
    if any(sp.issparse(X) for X in Xs):
        return sp.vstack(Xs)
    else:
        return np.vstack(Xs)


def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T)."""
    return np.dot(X.ravel(), Y.ravel())


def _sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)


def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
                    random_state=None):
    """NNDSVD algorithm for NMF initialization.
    Computes a good initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    Parameters
    ----------
    X : array, [n_samples, n_features]
        The data matrix to be decomposed.
    n_components : array, [n_components, n_features]
        The number of components desired in the approximation.
    variant : None | 'a' | 'ar'
        The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
        None: leaves the zero entries as zero
        'a': Fills the zero entries with the average of X
        'ar': Fills the zero entries with standard normal random variates.
        Default: None
    eps: float
        Truncate all values less then this in output to zero.
    random_state : numpy.RandomState | int, optional
        The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random
    Returns
    -------
    (W, H) :
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for 
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
	#Confirm that the matrix has all positive values
    check_non_negative(X, "NMF initialization")
	#parameter must meet one of these 3 options
    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")
	
	#take a random SVD using our data and n components (2 in this case) becomes U, S, H
    U, S, V = randomized_svd(X, n_components)
	#W(rows of X by 2), H(2 by columns of X) are the products of NMF, we use the same shapes as U and V
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
	#^What they said.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

	#we go from 1 to 2
    for j in range(1, n_components):
        #x is set to all the rows and the jth column of U, y is set to the jth row and all the columns of V
		x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
		#^again this person says it. x_p is the largest value in the x vector, same for y_p in y vector
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
		#absolute value of the minimum value in each vector to x_n and y_n
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
		#^this person is good!
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
		#define the multiplied norms
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
		#the norms multiplied tell us which way to converge. p or n. similarly we create new u, v and 
		#sigma values
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

		#add in a lbd which is the sqrt of the jth item of the S matrix * new sigma
		#(non negative diagonal values shape is same as X) = S matrix
        lbd = np.sqrt(S[j] * sigma)
		#Same way to create W's rows at the jth column
        W[:, j] = lbd * u
		#and H's column's at the jth row
        H[j, :] = lbd * v
	#Quick zero out of everything lower than zero in W and H
    W[W < eps] = 0
    H[H < eps] = 0

	#variant a
    if variant == "a":
		#avg is the mean of X
        avg = X.mean()
		#all zeros in W and H are now the avg
        W[W == 0] = avg
        H[H == 0] = avg
	#variant ar
    elif variant == "ar":
		#random state
        random_state = check_random_state(random_state)
        #avg is the mean of X
		avg = X.mean()
		#all zeros in W and H are now the absolute value of (avg * a random  number between 0 and the 
		#number of entries in W and H equal to zero) / 100
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

	#results in a W and an H. This works because it runs at least 200 times and picks the best W and H
	#as defined by reconstruction error (lower values means the froenbius norm is closer to zero) meaning
	#the lower the error the closer the product of W and H is to X
    return W, H