from __future__ import division, print_function

import numpy
from multiprocessing import Pool
from multiprocessing import cpu_count

LN_TEN = numpy.log(10.0)

'''
In the intensity function, we are multiplying the rate with the Probability distributoin of m_1 and m_2.

'''
def intensity(
        indiv_params, pop_params,
        aux_info,
        M_max=None,
        **kwargs
        ):
    import prob 
    import numpy as np
    m_1, m_2, ecc = indiv_params.T  #adding ecc as a random variable into the model. it would be given in the input data files.   
    log_rate, alpha, m_min, m_max, sigma_ecc = pop_params #adding sigma_ecc into the population model as a parameters which will be our output.

    pdf_const = aux_info["pdf_const"]
    rate = aux_info["rate"]

    return (
        # powerlaw in mass, one-sided normal distribution in ecc.  Assume ecc must be < 1 for sanity, but not enforcing
       rate *  prob.joint_pdf(m_1, m_2, alpha, m_min, m_max, M_max, const=pdf_const)* np.exp(-0.5*(ecc/sigma_ecc)**2)*np.sqrt(2/np.pi)/sigma_ecc
    )
'''
In the fowlling function "expval_mc", we are making a setup to calculate the equation 6 in the Paper. The equation 6 is basically the integral which is calculated by the mcmc integral and defined at the very end of the "expval_mc" function.

The method is as follows:
    The equation 5 gives us the senstivity of the LIGO.
    after calculating the senstiivty V, we run the instrument on the specific sensitivity for a time T.
    Then, we multiply V and T, and take the integral for the product with rate over lambda which gives us the average number of detections.

Read Prob.py where we defined the probability distributions which we are using here. 


'''

def expval_mc(
        pop_params, aux_info,
        raw_interpolator=None,
        M_max=None,
        err_abs=1e-5, err_rel=1e-3,
        return_err=False,
        rand_state=None,
        **kwargs
        ):
    import numpy as np
    import prob
    import mc

    log_rate, alpha, m_min, m_max, sigma_ecc = pop_params
    rate = aux_info["rate"]

    def p(N): #random number genrator from the pdf defined in the prob.py file and modifed above
        val= prob.joint_rvs(
            N,
            alpha, m_min, m_max, M_max,
            rand_state=rand_state,
            )
        ecc_vals = np.abs(np.random.normal(loc=0,scale=sigma_ecc,size=N))
        return np.c_[val,ecc_vals]

    def efficiency_fn(m1_m2,raw_interpolator=raw_interpolator):
        import numpy

        m1 = m1_m2[:,0]
        m2 = m1_m2[:,1]
        print(m1_m2.shape,m1_m2)
        return raw_interpolator(m1_m2[:,:2])

    I, err_abs, err_rel = mc.integrate_adaptive(
        p, efficiency_fn,
        err_abs=err_abs, err_rel=err_rel,
    )

    if return_err:
        return rate*I*230, err_abs, err_rel
    else:
        return rate*I*230

def VT_interp(m1_m2, raw_interpolator, **kwargs):
    import numpy

    m1 = m1_m2[:,0]
    m2 = m1_m2[:,1]

    return raw_interpolator(m1, m2)

'''
The above efficiency function is the integral which is solving the equation 6 in the paper. As per my understanding the above intensity function is just the product p*R.
To find the solution of equation 6 Dan used the mc integral and random numbers are genereated by p(N) from the joint pdf. The required function is defined in efficiency_fn.
but is efficiency_fn  actually a function or just the numbers as well?
Equation 6 = VT * R * P , I think VT is also the function of lambda and we are integrating over lambda.

IN the following function, we are defining our priors.
'''

def log_prior_pop(
        pop_params, aux_info,
        M_max=None,
        log_rate_min=None, log_rate_max=None,
        alpha_min=None, alpha_max=None,
        m_min_min=None, m_min_max=None,
        m_max_min=None, m_max_max=None,
        sigma_ecc_min=0, sigma_ecc_max=0.5,
        sigma_ecc_scale = "uniform",
        log_rate_scale="uniform", alpha_scale="uniform",
        m_min_scale="uniform", m_max_scale="uniform",
        **kwargs
    ):
    import numpy

    log_rate, alpha, m_min, m_max, sigma_ecc = pop_params

    log_prior = 0.0

    if log_rate_scale == "uniform": # I do need the explanation of this if not syntax.?????????
        if not (log_rate_min <= log_rate <= log_rate_max):
            return -numpy.inf
    else:
        raise NotImplementedError


    if alpha_scale == "uniform":
        if not (alpha_min <= alpha <= alpha_max):
            return -numpy.inf
    else:
        raise NotImplementedError


    if m_min_scale == "uniform":
        if not (m_min_min <= m_min <= m_min_max):
            return -numpy.inf
    else:
        raise NotImplementedError


    if m_max_scale == "uniform":
        if not (m_max_min <= m_max <= m_max_max):
            return -numpy.inf
    else:
        raise NotImplementedError


    if m_min >= m_max:
        return -numpy.inf


    if m_min + m_max > M_max:
        return -numpy.inf

    if sigma_ecc_scale == "uniform":
        if not (sigma_ecc_min <= sigma_ecc <= sigma_ecc_max):
            return -numpy.inf
    else:
        raise NotImplementedError

    return log_prior


def init_uniform( 
        nwalkers, nevents,
        fixed_log_rate, fixed_alpha,
        fixed_m_min, fixed_m_max,
        log_rate_min, log_rate_max,
        alpha_min, alpha_max,
        m_min_min, m_min_max,
        m_max_min, m_max_max,
        sigma_ecc_min, sigma_ecc_max,
        log_rate_scale="uniform",
        alpha_scale="uniform",
        m_min_scale="uniform",
        m_max_scale="uniform",
        rand_state=None,
    ):
    import numpy
    import utils

    rand_state = utils.check_random_state(rand_state)

    columns = []

    if fixed_log_rate is None:
        if log_rate_scale == "uniform":
            log_rate = rand_state.uniform(log_rate_min, log_rate_max, nwalkers)
        else:
            raise NotImplementedError

        columns.append(log_rate)

    if fixed_alpha is None:
        if alpha_scale == "uniform":
            alpha = rand_state.uniform(alpha_min, alpha_max, nwalkers)
        else:
            raise NotImplementedError

        columns.append(alpha)

    if fixed_m_min is None:
        if m_min_scale == "uniform":
            m_min = rand_state.uniform(m_min_min, m_min_max, nwalkers)
        else:
            raise NotImplementedError

        columns.append(m_min)

    if fixed_m_max is None:
        if m_max_scale == "uniform":
            m_max = rand_state.uniform(m_max_min, m_max_max, nwalkers)
        else:
            raise NotImplementedError

        columns.append(m_max)

    sigma_ecc = rand_state.uniform(0,0.5, nwalkers)
    columns.append(sigma_ecc)

    return numpy.column_stack(tuple(columns))


def _get_args(raw_args):
    import argparse, os

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "events",
        nargs="*",
        help="List of posterior sample files, one for each event.",
    )
    parser.add_argument(
        "VTs",
        help="HDF5 file containing VTs.",
    )
    parser.add_argument(
        "posterior_output",
        help="HDF5 file to store posterior samples in.",
    )

    parser.add_argument(
        "--fixed-log-rate",
        type=float,
        help="Constant to fix log_rate to (optional).",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        help="Constant to fix alpha to (optional).",
    )
    parser.add_argument(
        "--fixed-m-min",
        type=float,
        help="Constant to fix m_min to (optional).",
    )
    parser.add_argument(
        "--fixed-m-max",
        type=float,
        help="Constant to fix m_max to (optional).",
    )

    parser.add_argument(
        "--n-walkers",
        default=None, type=int,
        help="Number of walkers to use, defaults to twice the number of "
             "dimensions.",
    )
    parser.add_argument(
        "--n-samples",
        default=100, type=int,
        help="Number of MCMC samples per walker.",
    )
    parser.add_argument(
        "--n-threads",
        default=1, type=int,
        help="Number of threads to use in MCMC.",
    )

    parser.add_argument(
        "--log-rate-prior",
        default="uniform",
        choices=["uniform"],
        help="Type of prior used for log rate.",
    )
    parser.add_argument(
        "--log-rate-initial-cond",
        default="uniform",
        choices=["uniform"],
        help="Type of initial condition used for log rate.",
    )
    parser.add_argument(
        "--log-rate-min",
        type=float, default=-5.0,
        help="Minimum log10 rate allowed.",
    )
    parser.add_argument(
        "--log-rate-max",
        type=float, default=5.0,
        help="Maximum log10 rate allowed.",
    )

    parser.add_argument(
        "--alpha-prior",
        default="uniform",
        choices=["uniform"],
        help="Type of prior used for power law index 'alpha'.",
    )
    parser.add_argument(
        "--alpha-initial-cond",
        default="uniform",
        choices=["uniform"],
        help="Type of initial condition used for power law index 'alpha'.",
    )
    parser.add_argument(
        "--alpha-min",
        type=float, default=-5,
        help="Minimum alpha allowed.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float, default=+5,
        help="Maximum alpha allowed.",
    )

    parser.add_argument(
        "--m-min-prior",
        default="uniform",
        choices=["uniform"],
        help="Type of prior used for minimum component mass 'm_min'.",
    )
    parser.add_argument(
        "--m-min-initial-cond",
        default="uniform",
        choices=["uniform"],
        help="Type of initial condition used for minimum component mass "
             "'m_min'.",
    )
    parser.add_argument(
        "--m-min-min",
        type=float, default=1.0,
        help="Minimum m_min allowed.",
    )
    parser.add_argument(
        "--m-min-max",
        type=float, default=20.0,
        help="Maximum m_min allowed.",
    )

    parser.add_argument(
        "--m-max-prior",
        default="uniform",
        choices=["uniform"],
        help="Type of prior used for maximum component mass 'm_max'.",
    )
    parser.add_argument(
        "--m-max-initial-cond",
        default="uniform",
        choices=["uniform"],
        help="Type of initial condition used for maximum component mass "
             "'m_max'.",
    )
    parser.add_argument(
        "--m-max-min",
        type=float, default=30.0,
        help="Minimum m_max allowed.",
    )
    parser.add_argument(
        "--m-max-max",
        type=float, default=100.0,
        help="Maximum m_max allowed.",
    )

    parser.add_argument(
        "--mass-prior",
        default="uniform",
        choices=["uniform"],
        help="Type of prior used for component masses.",
    )
    parser.add_argument(
        "--total-mass-max",
        type=float, default=100.0,
        help="Maximum total mass allowed.",
    )

    parser.add_argument(
        "--mc-err-abs",
        type=float, default=1e-5,
        help="Allowed absolute error for Monte Carlo integrator.",
    )
    parser.add_argument(
        "--mc-err-rel",
        type=float, default=1e-3,
        help="Allowed relative error for Monte Carlo integrator.",
    )

    parser.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed.",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Use verbose output.",
    )

    return parser.parse_args(raw_args)


def _main(raw_args=None):
    import sys
    import six
    import h5py
    import numpy

    import vt  # not in pop models anymore, this is old material
    import mcmc # copy over
    from prob import param_names, ndim_pop

    # add eccentricity as param
    param_names += ['sigma_ecc']

    if raw_args is None: # I need to know the synatx of main,,, like in the main function, we have said that raw_args=None which is fixed now and then we are adding arg????
        raw_args = sys.argv[1:]

    cli_args = _get_args(raw_args)
    print(cli_args)
    rand_state = numpy.random.RandomState(cli_args.seed)

    M_max = cli_args.total_mass_max

    assert M_max is not None

    ncpu = cpu_count()
    print("{0} CPUs".format(ncpu))

    constants = {}
    if cli_args.fixed_log_rate is not None:
        constants["log_rate"] = cli_args.fixed_log_rate
    if cli_args.fixed_alpha is not None:
        constants["alpha"] = cli_args.fixed_alpha
    if cli_args.fixed_m_min is not None:
        constants["m_min"] = cli_args.fixed_m_min
    if cli_args.fixed_m_max is not None:
        constants["m_max"] = cli_args.fixed_m_max

    # TODO: open tabular file for events
    data_posterior_samples = []
    #print(data_posterior_samples)
    for event_fname in cli_args.events:
        data_table = numpy.genfromtxt(event_fname, names=True)
        m1_m2_ecc = numpy.column_stack(    #how it's reading the file and how we are telling him to look for specific data file. 
            (data_table["m1_source"], data_table["m2_source"], data_table["ecc"]),
        )
        data_posterior_samples.append(m1_m2_ecc)

        del data_table

    n_events = len(cli_args.events)

    # Compute array of priors for each posterior sample, so that we are doing
    # a Monte Carlo integral over the likelihood instead of the posterior, when
    # computing the event-based terms in the full MCMC.
    # If the prior is uniform, then no re-weighting is required, so we just set
    # it to ``None``, and ``pop_models.mcmc.run_mcmc`` takes care of the rest.
    if cli_args.mass_prior == "uniform":
        prior = None
    else:
        raise NotImplementedError

    # Load in <VT>'s.
    with h5py.File(cli_args.VTs, "r") as VTs:
        raw_interpolator = vt.interpolate_hdf5(VTs)

    # Determine number of dimensions for MCMC
    ndim = ndim_pop+1 - len(constants)
    # Set number of walkers for MCMC. If already provided use that, otherwise
    # use 2*ndim, which is the minimum allowed by the sampler.
    n_walkers = 2*ndim if cli_args.n_walkers is None else cli_args.n_walkers
    # print (ndim,n_walkers)
    # Initialize walkers
    init_state = init_uniform(
        n_walkers, n_events,
        cli_args.fixed_log_rate, cli_args.fixed_alpha,
        cli_args.fixed_m_min, cli_args.fixed_m_max,
        cli_args.log_rate_min, cli_args.log_rate_max,
        cli_args.alpha_min, cli_args.alpha_max,
        cli_args.m_min_min, cli_args.m_min_max,
        cli_args.m_max_min, cli_args.m_max_max,
        log_rate_scale=cli_args.log_rate_initial_cond,
        sigma_ecc_min=0, sigma_ecc_max=0.5,
        alpha_scale=cli_args.alpha_initial_cond,
        m_min_scale=cli_args.m_min_initial_cond,
        m_max_scale=cli_args.m_max_initial_cond,
        rand_state=rand_state,
    )


    # Create file to store posterior samples.
    # Fails if file already exists, to avoid accidentally deleting precious
    # samples.
    # TODO: Come up with a mechanism to resume from last walker positions if
    #   file exists.
    # TODO: Periodically update a counter indicating the number of samples
    #   drawn so far. This way if it's interrupted, we know which samples can
    #   be trusted, and which might be corrupted or just have the contents of
    #   previous memory.
    with h5py.File(cli_args.posterior_output, "w-") as posterior_output:
        # Store initial position.
        posterior_output.create_dataset(
            "init_pos", data=init_state,
        )
        # Create empty arrays for storing walker position and log_prob.
        posterior_pos = posterior_output.create_dataset(
            "pos", (cli_args.n_samples, n_walkers, ndim),
        )
        posterior_log_prob = posterior_output.create_dataset(
            "log_prob", (cli_args.n_samples, n_walkers),
        )

        # Store constants
        for name, value in six.iteritems(constants):
            posterior_output.attrs[name] = value
        # Store limits
        posterior_output.attrs["log_rate_min"] = cli_args.log_rate_min
        posterior_output.attrs["log_rate_max"] = cli_args.log_rate_max
        posterior_output.attrs["alpha_min"] = cli_args.alpha_min
        posterior_output.attrs["alpha_max"] = cli_args.alpha_max
        posterior_output.attrs["m_min_min"] = cli_args.m_min_min
        posterior_output.attrs["m_min_max"] = cli_args.m_min_max
        posterior_output.attrs["m_max_min"] = cli_args.m_max_min
        posterior_output.attrs["m_max_max"] = cli_args.m_max_max

        args = []
        kwargs = {
#            "efficiency_fn": VT_interp,
            "raw_interpolator": raw_interpolator,
            "M_max": M_max,
            "log_rate_min": cli_args.log_rate_min,
            "log_rate_max": cli_args.log_rate_max,
            "alpha_min": cli_args.alpha_min,
            "alpha_max": cli_args.alpha_max,
            "m_min_min": cli_args.m_min_min,
            "m_min_max": cli_args.m_min_max,
            "m_max_min": cli_args.m_max_min,
            "m_max_max": cli_args.m_max_max,
            "log_rate_scale": cli_args.log_rate_prior,
            "alpha_scale": cli_args.alpha_prior,
            "m_min_scale": cli_args.m_min_prior,
            "m_max_scale": cli_args.m_max_prior,
            "err_abs": cli_args.mc_err_abs,
            "err_rel": cli_args.mc_err_rel,
            "rand_state": rand_state,
        }

        pool=None;
        if True: #with None as pool:
        
         mcmc.run_mcmc(
            intensity, expval_mc, data_posterior_samples,
            log_prior_pop,
            init_state,
            param_names,
            constants=constants,
#            event_posterior_sample_priors=prior,
            args=args, kwargs=kwargs,
            before_prior_aux_fn=before_prior_aux_fn,
            after_prior_aux_fn=after_prior_aux_fn,
            out_pos=posterior_pos, out_log_prob=posterior_log_prob,
            nsamples=cli_args.n_samples,
            rand_state=rand_state,
            nthreads=cli_args.n_threads, pool=pool,
            runtime_sortingfn=None,
            dtype=numpy.float64,
            verbose=cli_args.verbose,
        )
# Functions which pre-compute quantities that are used at multiple steps
# in the MCMC, to reduce run time. These specifically compute the rate
# from the log10(rate), and the normalization factor for the mass
# distribution.
def before_prior_aux_fn(pop_params, **kwargs):
    log_rate, alpha, m_min, m_max, sigma_ecc = pop_params
    rate = 10 ** log_rate
    aux_info = {"rate": rate}

    return aux_info


def after_prior_aux_fn(
        pop_params, aux_info,
        M_max=None,
        **kwargs
    ):
    import prob

    log_rate, alpha, m_min, m_max, sigma_ecc = pop_params

    aux_info["pdf_const"] = prob.pdf_const(alpha, m_min, m_max, M_max)

    return aux_info


if __name__ == "__main__":
    _main()
