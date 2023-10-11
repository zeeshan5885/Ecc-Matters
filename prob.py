r"""
This module defines all of the probability density functions and random sampling
functions for the power law mass distribution model.
"""

from __future__ import division, print_function

import numpy as np
import scipy
from mpmath import hyp2f1

from prob_top import sample_with_cond
from utils_top import check_random_state

# Names of all (possibly free) parameters of this model, in the order they
# should appear in any array of samples (e.g., MCMC posterior samples).
param_names = ["log_rate", "alpha", "m_min", "m_max"]
# Number of (possibly free) parameters for this population model.
ndim_pop = len(param_names)


def powerlaw_rvs(N, alpha, x_min, x_max, rand_state=None):  # add ecc here
    r"""
    Draws ``N`` samples from a power law :math:`p(x) \propto x^{-\alpha}`, with
    support only betwen ``x_min`` and ``x_max``. Uses inverse transform
    sampling, drawing from the function

    .. math::
       \left((1-U) L + U H\right)^{1/\beta}

    where :math:`U` is uniformly distributed on (0, 1), and

    .. math::
       \beta = 1 - \alpha,
       L = x_{\mathrm{min}}^\beta,
       H = x_{\mathrm{max}}^\beta

    :param int N: Number of random samples to draw.

    :param float alpha: Power law index on :math:`p(x) \propto x^{-\alpha}`.

    :param float x_min: Lower limit of power law.

    :param float x_max: Upper limit of power law.

    :param np.random.RandomState rand_state: (optional) State for RNG.

    :return: array_like, shape (N,)
        Array of random samples drawn from distribution.
    """

    # Upgrade ``rand_state`` to an actual ``np.random.RandomState`` object,
    # if it isn't one already.
    rand_state = check_random_state(rand_state)

    # Power law index ``alpha`` only appears as ``1 - alpha`` in the equations,
    # so define this quantity as ``beta`` and use it henceforth.
    beta = 1 - alpha

    # Uniform random samples between zero and one.
    U = rand_state.uniform(size=N)

    # x_{min,max}^beta, which appear in the inverse transform equation
    L = np.power(x_min, beta)
    H = np.power(x_max, beta)

    # Compute the random samples.
    return np.power((1 - U) * L + U * H, 1.0 / beta)


def joint_rvs(
    N, alpha, m_min, m_max, M_max, rand_state=None
):  # same as in new_powerlaw_mcmc, but need to change it
    r"""
    Draws :math:`N` samples from the joint mass distribution :math:`p(m_1, m_2)`
    defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       \frac{m_1^{-\alpha}}{m_1 - m_{\mathrm{min}}}

    First draws samples from power law mass distribution :math:`p(m_1)`, then
    draws samples from the uniform distribution :math:`p(m_2 | m_1)`, and then
    rejects samples which do not satisfy the
    :math:`m_1 + m_2 \leq M_{\mathrm{max}}` constraint.
    """

    rand_state = check_random_state(rand_state)

    def rvs(N):  # need to generate rvs with three colums and third should be ecc
        """
        Draws ``m_1`` samples from a power law, and ``m_2`` samples uniformly
        between the minimum allowed mass and ``m_1``. Does not apply the total
        mass upper limit.
        """
        m_1 = powerlaw_rvs(N, alpha, m_min, m_max, rand_state=rand_state)
        m_2 = rand_state.uniform(m_min, m_1)

        return np.column_stack((m_1, m_2))

    def cond(m1_m2):
        """
        Given an array, where each row contains a pair ``(m_1, m_2)``, returns
        an array whose value is ``True`` when ``m_1 + m_2 <= M_max`` and
        ``False`` otherwise.
        """
        return np.sum(m1_m2, axis=1) <= M_max

    # Draw random samples from the "powerlaw in m_1, uniform in m_2"
    # distribution, and throw away samples not satisfying the total mass cutoff,
    # until precisely ``N`` samples are drawn.
    return sample_with_cond(rvs, shape=N, cond=cond)


def marginal_rvs(N, alpha, m_min, m_max, M_max, rand_state=None):
    r"""
    Draws :math:`N` samples from the marginal mass distribution :math:`p(m_1)`
    defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       m_1^{-\alpha} \,
       \frac{\min(m_1, M_{\mathrm{max}}-m_1) - m_{\mathrm{min}}}
            {m_1 - m_{\mathrm{min}}}

    Performs the sampling by drawing from :func:`joint_rvs` and discarding
    :math:`m_2`.
    """

    rand_state = check_random_state(rand_state)

    return joint_rvs(N, alpha, m_min, m_max, M_max, rand_state=rand_state)[:, 0]


def joint_pdf(
    m_1,
    m_2,
    alpha,
    m_min,
    m_max,
    M_max,
    const=None,
    out=None,
    where=True,
):
    r"""
    Computes the probability density for the joint mass distribution
    :math:`p(m_1, m_2)` defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       \frac{m_1^{-\alpha}}{m_1 - m_{\mathrm{min}}}

    Computes the normalization constant using :func:`pdf_const` if not provided
    by the ``const`` keyword argument.
    """

    # Ensure everything is a numpy array of the right shape.
    m_1, m_2 = np.broadcast_arrays(m_1, m_2)
    alpha, m_min, m_max = np.broadcast_arrays(alpha, m_min, m_max)

    m_1 = np.asarray(m_1)
    m_2 = np.asarray(m_2)
    alpha = np.asarray(alpha)
    m_min = np.asarray(m_min)
    m_max = np.asarray(m_max)

    S = m_1.shape
    T = alpha.shape
    TS = T + S

    out_type = m_1.dtype

    # Create a version of where which has shape ``T + S``.
    print(where)
    if where == True:
        where_TS = np.ones(TS, dtype=bool)
    else:
        where_TS = np.broadcast_to(where.T, S[::-1] + T[::-1]).T

    # Initialize output array.  Fill with zeros because we will only evaluate at
    # the support.  Alternatively use provided ``out`` array, and zero out
    # indices that we need to compute (marked by ``where``).
    if out is None:
        pdf = np.zeros(TS, dtype=out_type)
    else:
        pdf = out
        pdf[where_TS] = 0.0

    # Create transposed view of ``pdf`` for when we need operations to have
    # shape ``S[::-1]+T[::-1]`` instead of ``T+S``.
    pdf_T = pdf.T

    # Initialize a shape ``T+S`` float array to be reused.
    tmp_TS = np.empty(TS, dtype=np.float64)

    # Initialize two shape ``T+S`` index arrays, to be reused.
    tmp_TS_i1 = np.empty(TS, dtype=bool)
    tmp_TS_i2 = np.empty(TS, dtype=bool)

    # Create array of booleans determining which combinations of masses and
    # parameters do not correspond to zero probability and also have
    # ``where=True``.
    mmin_lt_m1 = np.less.outer(
        m_min,
        m_1,
        out=tmp_TS_i1,
        where=where_TS,
    )
    mmin_le_m2 = np.less_equal.outer(
        m_min,
        m_2,
        out=tmp_TS_i2,
        where=where_TS,
    )
    mmin_bound = np.logical_and(
        mmin_lt_m1,
        mmin_le_m2,
        out=tmp_TS_i1,
        where=where_TS,
    )
    del mmin_lt_m1, mmin_le_m2

    mmax_bound = np.greater_equal.outer(
        m_max,
        m_1,
        out=tmp_TS_i2,
        where=where_TS,
    )
    component_bounds = np.logical_and(
        mmin_bound,
        mmax_bound,
        out=tmp_TS_i1,
        where=where_TS,
    )
    del mmin_bound, mmax_bound, tmp_TS_i2

    mass_ordering = m_1 >= m_2
    Mmax_cutoff = m_1 + m_2 <= M_max
    Mmax_and_ordering = np.logical_and(
        mass_ordering,
        Mmax_cutoff,
        out=mass_ordering,
    )
    del mass_ordering, Mmax_cutoff

    i_eval = np.logical_and(
        component_bounds,
        Mmax_and_ordering,
        out=tmp_TS_i1,
        where=where_TS,
    )
    del component_bounds, Mmax_and_ordering
    i_eval = np.logical_and(
        i_eval,
        where_TS,
        out=tmp_TS_i1,
    )
    del tmp_TS_i1

    # Create transposed view of ``i_eval`` for when we need operations to have
    # shape ``S[::-1]+T[::-1]`` instead of ``T+S``.
    i_eval_T = i_eval.T

    # Compute the normalization constant if it has not been pre-computed.
    if const is None:
        const = pdf_const(alpha, m_min, m_max, M_max, where=where)

    # Compute the PDF at the indices specified by ``i_eval``.
    # Start with the powerlaw term, storing it in ``pdf``, then the subtraction
    # term that goes in the denominator, storing it in ``tmp_TS``.  Then divide
    # the two, storing the result in ``pdf``.  Finally multiply the normalizing
    # constants on to ``pdf``.
    powerlaw_term = np.power.outer(
        m_1.T,
        -alpha.T,
        out=pdf.T,
        where=i_eval.T,
    ).T
    denom_term = np.subtract.outer(
        m_1.T,
        m_min.T,
        out=tmp_TS.T,
        where=i_eval.T,
    ).T
    np.divide(
        powerlaw_term,
        denom_term,
        out=pdf,
        where=i_eval,
    )
    del powerlaw_term, denom_term, tmp_TS
    np.multiply(
        const.T,
        pdf.T,
        out=pdf.T,
        where=i_eval.T,
    )

    return pdf


def marginal_pdf(m1, alpha, m_min, m_max, M_max, const=None):
    r"""
    Computes the probability density for the marginal mass distribution
    :math:`p(m_1)` defined in :mod:`pop_models.powerlaw` as

    .. math::
       p(m_1, m_2) =
       C(\alpha, m_{\mathrm{min}}, m_{\mathrm{max}}, M_{\mathrm{max}}) \,
       m_1^{-\alpha} \,
       \frac{\min(m_1, M_{\mathrm{max}}-m_1) - m_{\mathrm{min}}}
            {m_1 - m_{\mathrm{min}}}

    Computes the normalization constant using :func:`pdf_const` if not provided
    by the ``const`` keyword argument.
    """

    ## Unoptimized version of this code.
    # for i, (alpha, m_min, m_max) in enumerate(zip(alphas, m_mins, m_maxs)):
    #     support = (m1 > m_min) & (m1 <= m_max)
    #     m1_support = m1[support]
    #
    #     if const is None:
    #         const = pdf_const(alpha, m_min, m_max, M_max)
    #
    #     pl_term = np.power(m1_support, -alpha)
    #     cutoff_term = (
    #         (np.minimum(m1_support, M_max-m1_support) - m_min) /
    #         (m1_support - m_min)
    #     )
    #
    #     pdf[i, support] = const * pl_term * cutoff_term

    alpha, m_min, m_max = np.broadcast_arrays(alpha, m_min, m_max)

    m1 = np.asarray(m1)
    alpha = np.asarray(alpha)
    m_min = np.asarray(m_min)
    m_max = np.asarray(m_max)

    S = m1.shape
    T = alpha.shape
    TS = T + S

    out_type = m1.dtype

    pdf = np.zeros(TS, dtype=out_type)

    tmp_TS_1 = np.empty(TS, dtype=out_type)
    tmp_TS_2 = np.empty(TS, dtype=out_type)
    tmp_S = np.empty(S, dtype=out_type)

    if const is None:
        const = pdf_const(alpha, m_min, m_max, M_max)
    const = np.asarray(const)

    # Index array containing ``True`` where the PDF has support, i.e., where
    # m_min < m_1 < m_max.
    support = np.less.outer(m_min, m1)
    np.logical_and.at(
        support,
        True,
        np.greater_equal.outer(m_max, m1),
    )

    # Store the powerlaw contribution to the probability density in ``pdf``.
    np.power.outer(
        m1,
        -alpha.T,
        out=pdf.T,
        where=support.T,
    )

    # Store min(m1, M_max-m1) in ``tmp_S``.
    np.subtract(M_max, m1, out=tmp_S)
    np.minimum.at(tmp_S, True, m1)

    # Subtract ``m_min`` from that and store it in ``tmp_TS_1``.
    # No longer need ``tmp_S`` after this.
    np.subtract.outer(tmp_S.T, m_min.T, out=tmp_TS_1.T, where=support.T)
    del tmp_S

    # Compute the denominator term ``m_1 - m_min`` and store it in ``tmp_TS_2``.
    np.subtract.outer(m1.T, m_min.T, out=tmp_TS_2.T, where=support.T)

    # Take the ratio of ``tmp_TS_1`` and ``tmp_TS_2``, overwriting the result of
    # ``tmp_TS_1``.  This gives the full cutoff term.  No longer need
    # ``tmp_TS_2``.
    np.divide(tmp_TS_1, tmp_TS_2, out=tmp_TS_1, where=support)
    del tmp_TS_2

    # Multiply the cutoff term onto the result.  No longer need any tmp arrays.
    np.multiply(pdf, tmp_TS_1, out=pdf, where=support)
    del tmp_TS_1

    # Multiply the normalizing constant onto the result.
    np.multiply(pdf.T, const.T, out=pdf.T, where=support.T)

    # Return the complete PDF.
    return pdf


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

    beta = 1 - alpha

    S = alpha.shape

    if out is None:
        result = np.empty(S, dtype=np.float64)
    else:
        result = out

    # Initialize temporary shape ``S`` array to hold booleans.
    tmp_i = np.zeros(S, dtype=bool)

    # Determine where the special case ``beta == 0`` occurs, and evaluate the
    # normalization constant there.
    special = np.equal(
        beta,
        0.0,
        out=tmp_i,
        where=where,
    )
    _pdf_const_special(
        m_min,
        m_max,
        M_max,
        out=result,
        where=special,
    )

    # Determine where the special case ``beta == 0`` does not occur, and
    # evaluate the normalization constant there.
    nonspecial = np.logical_not(
        special,
        out=tmp_i,
        where=where,
    )
    _pdf_const_nonspecial(
        beta,
        m_min,
        m_max,
        M_max,
        out=result,
        where=nonspecial,
    )

    return result


def _pdf_const_special(
    m_min,
    m_max,
    M_max,
    out=None,
    where=True,
):
    S = m_min.shape

    # Initialize temporary shape ``S`` array to hold booleans.
    tmp_i = np.zeros(S, dtype=bool)

    # Separately handle populations that are affected by the M_max cutoff and
    # those that are not.
    cutoff = np.greater(
        m_max,
        0.5 * M_max,
        out=tmp_i,
        where=where,
    )
    _pdf_const_special_cutoff(
        m_min,
        m_max,
        M_max,
        out=out,
        where=cutoff,
    )

    noncutoff = np.logical_not(
        cutoff,
        out=tmp_i,
        where=where,
    )
    _pdf_const_special_noncutoff(
        m_min,
        m_max,
        out=out,
        where=noncutoff,
    )

    return out


def _pdf_const_special_cutoff(
    m_min,
    m_max,
    M_max,
    out=None,
    where=True,
):
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


def _pdf_const_special_noncutoff(
    m_min,
    m_max,
    out=None,
    where=True,
):
    S = m_min.shape

    # Initialize temporary shape ``S`` array to hold floats.
    tmp = np.empty(S, dtype=np.float64)

    log_mmax = np.log(m_max, out=out, where=where)
    log_mmin = np.log(m_min, out=tmp, where=where)
    delta = np.subtract(log_mmax, log_mmin, out=out, where=where)
    del tmp

    return np.reciprocal(delta, out=out, where=where)


def _pdf_const_nonspecial(
    beta,
    m_min,
    m_max,
    M_max,
    eps=1e-7,
    out=None,
    where=True,
):
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
    integral = np.equal(
        beta.astype(np.int64),
        beta,
        out=tmp_i,
        where=where,
    )

    beta_neg = np.subtract(beta, eps, out=tmp2, where=integral)
    _pdf_const_nonspecial_nonintegral(
        beta_neg,
        m_min,
        m_max,
        M_max,
        out=out,
        where=integral,
    )

    beta_pos = np.add(beta, eps, out=tmp2, where=integral)
    _pdf_const_nonspecial_nonintegral(
        beta_pos,
        m_min,
        m_max,
        M_max,
        out=tmp1,
        where=integral,
    )
    del beta_neg, beta_pos, tmp2

    np.add(out, tmp1, out=out, where=where)
    np.multiply(0.5, out, out=out, where=where)

    # Now compute the normalization constant for the non-integral indices.
    # We'll negate and re-use the same integer array from before.
    nonintegral = np.logical_not(
        integral,
        out=tmp_i,
        where=where,
    )
    del integral

    _pdf_const_nonspecial_nonintegral(
        beta,
        m_min,
        m_max,
        M_max,
        out=out,
        where=nonintegral,
    )

    return out


def _pdf_const_nonspecial_nonintegral(
    beta,
    m_min,
    m_max,
    M_max,
    out=None,
    where=True,
):
    S = beta.shape

    # Initialize temporary shape ``S`` boolean array to all ``False``, as we
    # want to use ``where`` to only modify certain indices, and the ones left
    # un-modified need to be ``False``.
    tmp_i = np.zeros(S, dtype=bool)

    # Determine which indices need to be computed with the ``M_max`` cutoff
    # in effect, and compute them.
    cutoff = np.greater(
        m_max,
        0.5 * M_max,
        out=tmp_i,
        where=where,
    )
    _pdf_const_nonspecial_cutoff(
        beta,
        m_min,
        m_max,
        M_max,
        out=out,
        where=cutoff,
    )

    # Now compute the remaining terms.
    noncutoff = np.logical_not(
        cutoff,
        out=tmp_i,
        where=where,
    )
    _pdf_const_nonspecial_noncutoff(
        beta,
        m_min,
        m_max,
        out=out,
        where=noncutoff,
    )

    return out


def _pdf_const_nonspecial_cutoff(
    beta,
    m_min,
    m_max,
    M_max,
    out=None,
    where=None,
):
    if where is None:
        _, where = np.broadcast_arrays(out, where)

    beta_full, m_min_full, m_max_full = beta, m_min, m_max

    for i, _ in np.ndenumerate(beta_full):
        if not where[i]:
            continue
        beta, m_min, m_max = beta_full[i], m_min_full[i], m_max_full[i]

        A = (np.power(0.5 * M_max, beta) - np.power(m_min, beta)) / beta

        B1a = np.power(0.5 * M_max, beta) * hyp2f1(
            1, beta, 1 + beta, 0.5 * M_max / m_min
        )
        B1b = np.power(m_max, beta) * hyp2f1(1, beta, 1 + beta, m_max / m_min)
        B1 = (M_max - 2 * m_min) * (B1a - B1b) / m_min

        B2 = np.power(0.5 * M_max, beta) - np.power(m_max, beta)

        B = np.float64((B1 + B2).real) / beta

        out[i] = np.reciprocal(A + B)

    return out


def _pdf_const_nonspecial_noncutoff(
    beta,
    m_min,
    m_max,
    out=None,
    where=True,
):
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


def upper_mass_credible_region(
    quantile,
    alpha,
    m_min,
    m_max,
    M_max,
    n_samples=1000,
    m1_samples=None,
):
    if m1_samples is None:
        m1_samples = np.linspace(m_min, m_max, n_samples)

    f = marginal_pdf(m1_samples, alpha, m_min, m_max, M_max, const=1.0)[0]

    P = scipy.integrate.cumtrapz(f, m1_samples, initial=0.0)
    P /= P[-1]

    return m1_samples[P >= quantile][0]


def upper_mass_credible_region_detection_weighted(
    quantile,
    alpha,
    m_min,
    m_max,
    M_max,
    VT_from_m1_m2,
    min_samples_m2=10,
    max_samples_m2=1000,
    dm2_ideal=0.5,
    n_samples_m1=1000,
    m1_samples=None,
    dm1=None,
):
    if m1_samples is None:
        m1_samples, dm1 = np.linspace(
            m_min,
            m_max,
            n_samples_m1,
            retstep=True,
        )
        # Don't need to provide ``x`` for m1 integral, because we have a
        # constant and known ``dm1``.
        x_m1_int = None
    else:
        # Only provide ``x`` for m1 integral if ``dm1`` was not provided.
        if dm1 is None:
            x_m1_int = m1_samples
        else:
            x_m1_int = None

    f = np.empty_like(m1_samples)

    for i, m1 in enumerate(m1_samples):
        m2_max = min(m1, M_max - m1)
        m2_range = m2_max - m_min

        n_samples_m2 = m2_range / dm2_ideal

        if n_samples_m2 < min_samples_m2:
            n_samples_m2 = min_samples_m2
        elif n_samples_m2 > max_samples_m2:
            n_samples_m2 = max_samples_m2
        else:
            n_samples_m2 = int(np.ceil(n_samples_m2))

        m2_samples, dm2 = np.linspace(
            m_min,
            m2_max,
            n_samples_m2,
            retstep=True,
        )

        integrand = VT_from_m1_m2(m1, m2_samples) * joint_pdf(
            m1, m2_samples, alpha, m_min, m_max, M_max, const=1.0
        )

        f[i] = scipy.integrate.trapz(integrand, dx=dm2)

    P = scipy.integrate.cumtrapz(f, x=x_m1_int, dx=dm1, initial=0.0)
    P /= P[-1]

    return m1_samples[P >= quantile][0]
