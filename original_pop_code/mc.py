from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

import sklearn.utils as utils

def integrate_box(f, lower, upper, N, compute_error=False, random_state=None):
    """
    Compute the Monte Carlo integral
    """

    rng = utils.check_random_state(random_state)

    V = np.prod(np.subtract(upper, lower))

    X = np.column_stack(rng.uniform(lo, hi, N) for lo, hi in zip(lower, upper))
    fX = f(X)
    F = np.mean(fX)
    I = V * F

    if not compute_error:
        return I

    F2 = np.mean(np.power(fX, 2))
    dI = V * np.sqrt((F2 - F * F) / N)

    return I, dI


def integrate_adaptive(p, f, iter_max=2**16, iter_start=2**10, err_abs=None, err_rel=None, plot=False):
    """
    Compute an adaptive Monte Carlo integral of the form
      Integral p(x) * f(x) dx
    using p(x) to draw samples.

    To take advantage of vectorization, the function is evaluated in batches.
    Starts by evaluating f(x) ``iter_start`` times, and doubles the number of
    evaluations until convergence, or ``iter_max`` is reached. Convergence
    criteria are optional, and can be specified by ``err_abs`` for the maximum
    standard error, and ``err_rel`` for the maximum relative error.

    **Inputs**

    p: function(N) -> array of shape (N,)
        A function which samples from a probability distribution. Must take a
        single integer ``N`` as an input, and returns an array of ``N`` samples
        from the desired distribution.

    f: function(array of shape (N,)) -> array of shape (N,)
        The function to be integrated. Must accept as input the array produced
        by ``p``, and output an array of the same shape.

    iter_max: int, default = 2**16
        The maximum number of iterations to perform before ending the
        computation.

    iter_start: int, default = 2**10
        The initial number of iterations to perform, before testing for
        convergence. This number is doubled until ``iter_max`` is reached.

    err_abs: float, optional
        The maximum acceptable standard error for the integral to take. If not
        provided, this will not be used as a stopping criterion.

    err_rel: float, optional
        The maximum possible relative error for the integral to take. This is
        just the standard error divided by the value of the integral. If not
        provided, this will not be used as a stopping criterion.

    **Outputs**

    integral: float
        The estimated value of the integral.

    err_abs: float
        The estimated value of the standard error.

    err_rel: float
        The estimated value of the relative error.
    """

    # Define convergence test function. Instead of putting conditions -- which
    # always have the same value -- inside the function, we pull them outside
    # the function. This reduces the number of times we evaluate the conditions
    # to one.
    #
    # Convergence is defined as follows:
    #
    # If both ``err_abs`` and ``err_rel`` are provided, then convergence is
    # reached when the current absolute and relative errors are below those
    # thresholds.
    #
    # If only one of the two is not ``None``, convergence is reached when
    # the current error is below the given threshold.
    #
    # If neither of the error thresholds is provided, then the convergence test
    # always fails, and the routine will always end at the max number of
    # iterations.
    if err_abs is None and err_rel is None:

        def converged(err_abs_current, err_rel_current):
            return False

    elif err_abs is None:

        def converged(err_abs_current, err_rel_current):
            return err_rel_current < err_rel

    elif err_rel is None:

        def converged(err_abs_current, err_rel_current):
            return err_abs_current < err_abs

    else:

        def converged(err_abs_current, err_rel_current):
            return (err_rel_current < err_rel) and (err_abs_current < err_abs)

    # -- he should have used lambdas

    # Initialize samples and integral accumulators.
    samples = 0
    new_samples = iter_start
    F = 0.0
    F2 = 0.0

    # If ``plot`` is a truthy value, perform diagnostic plots. Note, only if it
    # is precisely the value ``True`` is the plot made interactively. Otherwise
    # ``plot`` is assumed to refer to a file to save to, once the routine ends.
    if plot:
        fig, (ax_integral, ax_err_abs, ax_err_rel) = plt.subplots(3, sharex=True)

        for ax in (ax_integral, ax_err_abs, ax_err_rel):
            ax.set_xscale("log", basex=2)
            ax.grid()

        ax_err_rel.set_xlabel(r"$N\ \mathrm{samples}$")

        ax_integral.set_ylabel(r"$\mathrm{Integral}$")
        ax_err_abs.set_ylabel(r"$\mathrm{AbsErr}$")
        ax_err_rel.set_ylabel(r"$\mathrm{RelErr} [\%]$")

    # Iteratively improve MC integral, until either the convergence criteria
    # set by ``err_abs`` and ``err_rel`` are reached, or the maximum number of
    # iterations is reached.
    while samples < iter_max:
        # Draw samples from the probability distribution function.
        X = p(new_samples)

        # Evaluate the function at these samples.
        # Accumulate the sum of the result in F and the squared result in F2,
        # which will be used to calculate the integral and its uncertainty.
        value = f(X)
        F += np.sum(value)
        F2 += np.sum(value * value)

        # Update the number of samples appropriately.
        # If more samples are needed to converge, the number will be doubled.
        samples += new_samples
        new_samples = samples

        # Estimate the integral as the sample average of f(X) over all
        # iterations.
        I_current = F / samples
        # Estimate the absolute error (standard error) and the relative error
        # (standard error divided by the integral).
        err_abs_current = np.sqrt((F2 - F * F / samples) / (samples * (samples - 1.0)))
        err_rel_current = err_abs_current / I_current

        # If a plot was requested, add the new estimates of the integral and
        # the errors to the figure.
        if plot:
            ax_integral.scatter(samples, I_current)
            ax_err_abs.scatter(samples, err_abs_current)
            ax_err_rel.scatter(samples, err_rel_current * 100.0)

        # Test for convergence, and end the routine if it has been reached.
        if converged(err_abs_current, err_rel_current):
            break

    # If a plot was requested, and was not literally ``True``, then we interpret
    # ``plot`` as a filename to save the plot to. Otherwise, we show the plot,
    # so long as we are not in interactive mode, in which case it should already
    # be visible.
    if plot:
        if not plt.isinteractive():
            plt.show(fig)
        else:
            fig.savefig(plot)

    return I_current, err_abs_current, err_rel_current
