# TODO: rename "data likelihood" to "sample likelihood"
from __future__ import division, print_function

import sys

import emcee
import numpy as np


def run_mcmc(intensity_fn,
             expval_fn,
             data_likelihood_samples,
             log_prior_fn,
             init_state,
             param_names,
             constants=None,
             data_likelihood_weights=None,
             args=None,
             kwargs=None,
             before_prior_aux_fn=None,
             after_prior_aux_fn=None,
             out_pos=None,
             out_log_prob=None,
             nsamples=100,
             rand_state=None,
             debug_log_prob=False,
             nthreads=1,
             pool=None,
             runtime_sortingfn=None,
             verbose=False,
             dtype=np.float64):
    """

    :param function intensity_fn:

    :param function expval_fn:

    :param array_like data_likelihood_samples:

    :param array_like init_state:

    :param list param_names:

    :param dict constants: (optional)

    :param array_like data_likelihood_weights: (optional)

    :param list args: (optional)

    :param dict kwargs: (optional)

    :param function before_prior_aux_fn: (optional)

    :param function after_prior_aux_fn: (optional)

    :param array_like out_pos: (optional)

    :param array_like out_log_prob: (optional)

    :param int nsamples: (default 100)

    :param np.random.RandomState rand_state: (optional)

    :param bool debug_log_prob: (optional)

    :param int nthreads: (optional)

    :param multiprocessing.Pool pool: (optional)

    :param function runtime_sortingfn: (optional)

    :param bool verbose: (optional)

    :param type dtype: (optional)

    :return: array_like, shape (n_samples, n_walkers, n_params)
        MCMC chain for each ensemble walker. Element ``[i,j,k]`` is the value of
        the ith sample in the chain, for the jth walker, for the kth free
        parameter.

    :return: array_like, shape (n_samples, n_walkers)
        Values of the (non-normalized) log-posterior function corresponding to
        each step in the MCMC chain for each ensemble walker. Element ``[i,j]``
        is the log-posterior for the ith sample in the chain, for the jth
        walker.
    """

    # If no args or kwargs provided, set as empty list/dict
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # If no constants provides, set as empty dict
    if constants is None:
        constants = {}

    # Count the number of free parameter dimensions
    ndim = len(param_names) - len(constants)

    # Count the number of walkers
    nwalkers = len(init_state)
    # Count the number of individual events
    nindiv = len(data_likelihood_samples)

    # We're going to iterate over the samples and weights together, so if there
    # are no weights, we at least need to make it into a list of the proper
    # size.
    if data_likelihood_weights is None:
        data_likelihood_weights = [None for _ in data_likelihood_samples]

    # Ensure samples all have same dimensionality
    ndim_indiv = None
    for samples in data_likelihood_samples:
        S, D = np.shape(samples)

        if ndim_indiv is None:
            ndim_indiv = D
        assert ndim_indiv == D

    # Ensure number of dimensions equals the number of dimensions in the initial
    # state.
    assert ndim == len(init_state[0])

    # Initialize output arrays if not provided.
    # Otherwise check provided arrays have proper shape.
    if out_pos is None:
        out_pos = np.empty((nsamples, nwalkers, ndim), dtype=dtype)
    else:
        assert np.shape(out_pos) == (nsamples, nwalkers, ndim)

    if out_log_prob is None:
        out_log_prob = np.empty((nsamples, nwalkers), dtype=dtype)
    else:
        assert np.shape(out_log_prob) == (nsamples, nwalkers)

    if debug_log_prob:
        return log_prob

    sampler_args = (param_names, constants, intensity_fn, expval_fn, log_prior_fn, data_likelihood_samples,
                    data_likelihood_weights, before_prior_aux_fn, after_prior_aux_fn, args, kwargs)

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_prob,
                                    args=sampler_args,
                                    threads=nthreads,
                                    pool=pool,
                                    runtime_sortingfn=runtime_sortingfn)
    sample_iter = sampler.sample(init_state, iterations=nsamples, rstate0=rand_state)

    if verbose:
        progress_pct = 0

        def display_progress(p, s):
            print(f"Progress: {p}%; Samples: {s}", file=sys.stderr)

        display_progress(progress_pct, 0)

    for i, result in enumerate(sample_iter):
        pos = result[0]
        log_post = result[1]

        out_pos[i] = pos
        out_log_prob[i] = log_post

        if verbose:
            new_progress_pct = i / nsamples * 100
            if new_progress_pct >= progress_pct + 1:
                progress_pct = int(new_progress_pct)
                display_progress(progress_pct, i)

    return out_pos, out_log_prob


def log_prob(params_free, param_names, constants, intensity_fn, expval_fn, log_prior_fn, data_likelihood_samples,
             data_likelihood_weights, before_prior_aux_fn, after_prior_aux_fn, args, kwargs):
    params = get_params(params_free, constants, param_names)

    if before_prior_aux_fn is not None:
        aux_info = before_prior_aux_fn(params, *args, **kwargs)
    else:
        aux_info = None

    log_pi = log_prior_fn(params, aux_info, *args, **kwargs)

    if np.isfinite(log_pi):
        if after_prior_aux_fn is not None:
            aux_info = after_prior_aux_fn(params, aux_info, *args, **kwargs)

        log_events_contribution = 0.0
        iterables = zip(data_likelihood_samples, data_likelihood_weights)
        for samples, weights in iterables:
            intensity = intensity_fn(samples, params, aux_info, *args, **kwargs)

            if weights is not None:
                intensity *= weights

            log_events_contribution += np.log(np.mean(intensity))

        mean = expval_fn(params, aux_info, *args, **kwargs)

        log_prob = log_pi + log_events_contribution - mean

        if np.isfinite(log_prob):
            return log_prob

    return -np.inf


def get_params(variables, constants, names):
    """ """
    if len(variables) + len(constants) != len(names):
        raise ValueError(
            f"Incorrect number of variables and constants. Expected {len(names)}, but got {len(variables) + len(constants)}."
        )

    params = []
    i = 0

    for name in names:
        if name in constants:
            param = constants[name]
        else:
            param = variables[i]
            i += 1

        params.append(param)

    return params
