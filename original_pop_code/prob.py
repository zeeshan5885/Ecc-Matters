r"""
This module defines all of the probability density functions and random sampling
functions for the power law mass distribution model.
"""

from __future__ import division, print_function

import numpy as np
import scipy.integrate
import scipy.special
from mpmath import hyp2f1

from prob_top import sample_with_cond
import sklearn.utils as utils

# Names of all (possibly free) parameters of this model, in the order they
# should appear in any array of samples (e.g., MCMC posterior samples).
param_names = ["log_rate", "alpha", "m_min", "m_max"]
# Number of (possibly free) parameters for this population model. 
ndim_pop = len(param_names)

def pdf_const(alpha, m_min, m_max, M_max, out=None, where=True):
    r"""
    Computes the normalization constant
    :math:`C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}})`,
    according to the derivation given in [T1700479]_.

    .. [T1700479]
        Normalization constant in power law BBH mass distribution model,
        Daniel Wysocki and Richard O'Shaughnessy,
        `LIGO-T1700479 <https://dcc.ligo.org/LIGO-T1700479>`_
    """

    alpha, m_min, m_max = np.broadcast_arrays(alpha, m_min, m_max)
    alpha = np.asarray(alpha)
    m_min = np.asarray(m_min)
    m_max = np.asarray(m_max)

    beta = 1.0 - alpha

    S = alpha.shape

    if out is None:
        result = np.empty(S, dtype=np.float64)
    else:
        result = out

    # Initialize temporary shape ``S`` array to hold booleans.
    tmp_i = np.zeros(S, dtype=bool)

    # Determine where the special case ``beta == 0`` occurs, and evaluate the
    # normalization constant there.
    special = np.equal(beta, 0.0, out=tmp_i, where=where)
    _pdf_const_special(m_min, m_max, M_max, out=result, where=special)

    # Determine where the special case ``beta == 0`` does not occur, and
    # evaluate the normalization constant there.
    nonspecial = np.logical_not(special, out=tmp_i, where=where)
    _pdf_const_nonspecial(beta, m_min, m_max, M_max, out=result, where=nonspecial)

    return result


def _pdf_const_special(m_min, m_max, M_max, out=None, where=True):
    S = m_min.shape

    # Initialize temporary shape ``S`` array to hold booleans.
    tmp_i = np.zeros(S, dtype=bool)

    # Separately handle populations that are affected by the M_max cutoff and
    # those that are not.
    cutoff = np.greater(m_max, 0.5 * M_max, out=tmp_i, where=where)
    _pdf_const_special_cutoff(m_min, m_max, M_max, out=out, where=cutoff)

    noncutoff = np.logical_not(cutoff, out=tmp_i, where=where)
    _pdf_const_special_noncutoff(m_min, m_max, out=out, where=noncutoff)

    return out


def _pdf_const_special_cutoff(m_min, m_max, M_max, out=None, where=True):
    ## Un-optimized version of the code
    # A = np.log(0.5) + np.log(M_max) - np.log(m_min)

    # B1 = (
    #     (M_max - 2*m_min) *
    #     np.log((m_max - m_min) / (0.5*M_max - m_min))
    # )
    # B2 = (M_max - m_min) * np.log(0.5 * M_max / m_max)
    # B = (B1 + B2) / m_min

    # return np.reciprocal(A + B)

    # Create two temporary arrays with the same dimension as ``out``, in order
    # to efficiently hold intermediate results.
    tmp1 = np.empty_like(out)
    tmp2 = np.empty_like(out)

    # Pre-compute ``0.5*M_max``, as it will come up a few times.
    half_Mmax = 0.5 * M_max

    # Start by computing B1, and storing the result in out
    np.subtract(m_max, m_min, out=out, where=where)
    np.subtract(half_Mmax, m_min, out=tmp1, where=where)
    np.divide(out, tmp1, out=out, where=where)
    np.multiply(-2.0, m_min, out=tmp1, where=where)
    np.add(tmp1, M_max, out=tmp1, where=where)
    B1 = scipy.special.xlogy(tmp1, out, out=out, where=where)

    # Now compute B2, and store the result in tmp1, without touching ``out`` as
    # we'll need its value later.
    np.divide(half_Mmax, m_min, out=tmp1, where=where)
    np.subtract(M_max, m_min, out=tmp2, where=where)
    B2 = scipy.special.xlogy(tmp2, tmp1, out=tmp1, where=where)

    # Now compute B = (B1+B2) / m_min, storing the result in ``out``.
    # After this, ``tmp2`` is no longer needed.
    np.add(B1, B2, out=out, where=where)
    B = np.divide(out, m_min, out=out, where=where)
    del B1, B2, tmp2

    # Now compute A, storing the result in ``tmp1``.
    np.log(m_min, out=tmp1, where=where)
    A = np.subtract(np.log(half_Mmax), tmp1, out=tmp1, where=where)

    # Now compute the final result, C = 1 / (A+B), storing each step in ``out``.
    # ``tmp1`` will not be needed after the first operation.  We also won't need
    # the explicit references to ``A`` and ``B`` anymore, so we delete them to
    # ensure garbage collection is triggered on ``tmp1``, and for clarity.
    np.add(A, B, out=out, where=where)
    del A, B, tmp1

    C = np.reciprocal(out, out=out, where=where)

    return C


def _pdf_const_special_noncutoff(m_min, m_max, out=None, where=True):
    S = m_min.shape

    # Initialize temporary shape ``S`` array to hold floats.
    tmp = np.empty(S, dtype=np.float64)

    log_mmax = np.log(m_max, out=out, where=where)
    log_mmin = np.log(m_min, out=tmp, where=where)
    delta = np.subtract(log_mmax, log_mmin, out=out, where=where)
    del tmp

    return np.reciprocal(delta, out=out, where=where)


def _pdf_const_nonspecial(beta, m_min, m_max, M_max, eps=1e-7, out=None, where=True):
    S = beta.shape

    # Initialize two temporary shape ``S`` arrays to hold floats when performing
    # averaging operation for integral ``beta``.  Also initialize temporary
    # shape ``S`` boolean array to all ``False``, as we want to use ``where`` to
    # only modify certain indices, and the ones left un-modified need to be
    # ``False``.
    tmp1 = np.zeros(S, dtype=np.float64)
    tmp2 = np.zeros(S, dtype=np.float64)
    tmp_i = np.zeros(S, dtype=bool)

    # Compute the indices where ``beta`` is an integer.
    # Then, for each of those indices, compute the normalization constant using
    # both ``beta+eps`` and ``beta-eps``, storing the results in ``out`` and
    # ``tmp1``.  Then add the two results into ``out``, and take half that to
    # get the arithmetic mean.  The ``beta+eps`` and ``beta-eps`` terms will be
    # stored in ``tmp2``, so we can free that memory once they're no longer
    # needed.
    integral = np.equal(beta.astype(np.int64), beta, out=tmp_i, where=where)

    beta_neg = np.subtract(beta, eps, out=tmp2, where=integral)
    _pdf_const_nonspecial_nonintegral(beta_neg, m_min, m_max, M_max, out=out, where=integral)

    beta_pos = np.add(beta, eps, out=tmp2, where=integral)
    _pdf_const_nonspecial_nonintegral(beta_pos, m_min, m_max, M_max, out=tmp1, where=integral)
    del beta_neg, beta_pos, tmp2

    np.add(out, tmp1, out=out, where=where)
    np.multiply(0.5, out, out=out, where=where)

    # Now compute the normalization constant for the non-integral indices.
    # We'll negate and re-use the same integer array from before.
    nonintegral = np.logical_not(integral, out=tmp_i, where=where)
    del integral

    _pdf_const_nonspecial_nonintegral(beta, m_min, m_max, M_max, out=out, where=nonintegral)

    return out


def _pdf_const_nonspecial_nonintegral(beta, m_min, m_max, M_max, out=None, where=True):
    S = beta.shape

    # Initialize temporary shape ``S`` boolean array to all ``False``, as we
    # want to use ``where`` to only modify certain indices, and the ones left
    # un-modified need to be ``False``.
    tmp_i = np.zeros(S, dtype=bool)

    # Determine which indices need to be computed with the ``M_max`` cutoff
    # in effect, and compute them.
    cutoff = np.greater(m_max, 0.5 * M_max, out=tmp_i, where=where)
    _pdf_const_nonspecial_cutoff(beta, m_min, m_max, M_max, out=out, where=cutoff)

    # Now compute the remaining terms.
    noncutoff = np.logical_not(cutoff, out=tmp_i, where=where)
    _pdf_const_nonspecial_noncutoff(beta, m_min, m_max, out=out, where=noncutoff)

    return out


def _pdf_const_nonspecial_cutoff(beta, m_min, m_max, M_max, out=None, where=None):
    if where is None:
        _, where = np.broadcast_arrays(out, where)

    beta_full, m_min_full, m_max_full = beta, m_min, m_max

    for i, _ in np.ndenumerate(beta_full):
        if not where[i]:
            continue
        beta, m_min, m_max = beta_full[i], m_min_full[i], m_max_full[i]

        A = (np.power(0.5 * M_max, beta) - np.power(m_min, beta)) / beta

        B1a = np.power(0.5 * M_max, beta) * hyp2f1(1, beta, 1 + beta, 0.5 * M_max / m_min)
        B1b = np.power(m_max, beta) * hyp2f1(1, beta, 1 + beta, m_max / m_min)
        B1 = (M_max - 2 * m_min) * (B1a - B1b) / m_min

        B2 = np.power(0.5 * M_max, beta) - np.power(m_max, beta)

        B = np.float64((B1 + B2).real) / beta

        out[i] = np.reciprocal(A + B)

    return out


def _pdf_const_nonspecial_noncutoff(beta, m_min, m_max, out=None, where=True):
    ## Un-optimized version of the code
    # return np.reciprocal(
    #     (np.power(m_max, beta) - np.power(m_min, beta)) / beta
    # )

    S = beta.shape

    # Initialize temporary shape ``S`` array to hold floats.
    tmp = np.zeros(S, dtype=np.float64)

    # Compute ``m_max**beta`` and ``m_min**beta``, and store them in ``out`` and
    # ``tmp``, respectively.
    np.power(m_max, beta, out=out, where=where)
    np.power(m_min, beta, out=tmp, where=where)

    # Perform the rest of the operations overwriting ``out``.  Can free ``tmp``
    # right after the first operation.
    np.subtract(out, tmp, out=out, where=where)
    del tmp
    np.divide(out, beta, out=out, where=where)
    np.reciprocal(out, out=out, where=where)

    return out

