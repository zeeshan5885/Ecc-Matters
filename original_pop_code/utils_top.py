from __future__ import division, print_function


# Taken from scikit-learn
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    import numbers
    import numpy as np

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)




def quantile(a, q, axis=None, weights=None, out=None,
             overwrite_input=False, interpolation='linear', keepdims=False):
    """
    Compute the qth quantile of the data along the specified axis,
    where q = [0, 1].

    Returns the qth quantiles(s) of the array elements.

    NOTE: This was taken from a yet-to-be-accepted pull request to numpy.
    Expect this to be merged with numpy at some point.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute, which must be between 0 and 100 inclusive.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array. A sequence of axes is supported since
        version 1.9.0.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis), or such that
        weights.ndim == a.ndim and that `weights` and `a` are broadcastable.
        If `weights=None`, then all data in `a` are assumed to have a weight
        equal to one.

        Weights cannot be:
        1.) negative
        2.) sum to 0
        However, they can be
        1.) 0, as long as (2) above is not violated
        2.) less than 1.  In this case, all weights are re-normalized by
            the lowest non-zero weight prior to computation.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow use of memory of input array `a`
        calculations. The input array will be modified by the call to
        `percentile`. This will save memory when you do not need to
        preserve the contents of the input array. In this case you
        should not make any assumptions about the contents of the input
        `a` after this function completes -- treat it as undefined.
        Default is False. If `a` is not already an array, this parameter
        will have no effect as `a` will be converted to an array
        internally regardless of the value of this parameter.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired quantile lies between two data points
        ``i < j``:
            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.

        .. versionadded:: 1.9.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

        .. versionadded:: 1.9.0

    Returns
    -------
    quantile : scalar or ndarray
        If `a` is 1-D, `q` is a single quantile, and `axis=None`,
        then the result is a scalar.
        First axis of the result corresponds to the percentiles.
        The other axes are the axes that remain after the reduction of `a`.
        If `a` is of type ``int`` and no interpolation between data points is
        required, then the output data-type is also ``int``.  Otherwise, the
        output data-type is ``float64``.  If `out` is specified, that array is
        returned instead.

    See Also
    --------
    average, percentile, nanpercentile, mean, median

    Notes
    -----
    Given a vector ``V`` of length ``N``, the ``q``-th quantile of
    ``V`` is the value ``q`` of the way from the minimum to the
    maximum in a sorted copy of ``V``. The values and distances of
    the two nearest neighbors as well as the `interpolation` parameter
    will determine the percentile if the normalized ranking does not
    match the location of ``q`` exactly. This function is the same as
    the median if ``q=0.5``, the same as the minimum if ``q=0`` and the
    same as the maximum if ``q=1``.

    In the simplest case, where all weights are integers,
    the `weights` argument can be seen as repeating the data in the sample.
    For example, if a = [1,2,3] and weights = [1,2,3], this will be identical
    to the case where there are no weights and a = [1,2,2,3,3,3].

    In this method, the computation of weighted quantile uses
    normalized cumulative weights, with some renormalization and edge-setting,
    to maintain consistency with np.percentile.

    The algorithm is illustrated here:

    a = np.array([3, 5, 4])
    q = [0.25, 0.5, 0.75]
    axis = 0
    weights = [1, 2, 3]

    The np.percentile way, after expansion:

    value       3   4   4   4   5   5
    cum_prob    0  .2  .4  .6  .8   1
    q=0.25            4
    q=0.5                 4
    q=0.75                    4.75

    returns [4, 4, 4.75]

    This method computes the cumulative probability bands associated with each
    value, based on the weights:

    value               3           4           5
    weight              1           3           2  (total = 6)
    cum. weight         .17         .67         1
    prob_band         0 - .17   .33 - .67   .83 - 1  (.17-.33 and .67-.83 are
                                                      transitions)
    w slice             .17         .17         .17  (diff in upper bound,
                                                      divided by weight)

    lower_bound         0           .33         .83
    upper_bound         .17         .67         1

    # now substract the bound values by the left-most w-slice value

    new_lower_bd        0           .17         .67
    new_upper_bd        0           .5          .83  --> used to renormalize

    renormalized_lower  0           .2          .8
    renormalized_upper  0           .6          1

    new line up         3       3       4       4       5       5
    quantile bounds     0       0       .2      .6      .8      1

    q=0.25                                 4
    q=0.5                                     4
    q=0.75                                          4.75

    returns [4, 4, 4.75]

    Examples
    --------
    >>> ar = np.arange(6).reshape(2,3)
    >>> ar
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.quantile(ar, q=0.5, weights=np.ones(6).reshape(2,3))
    2.5
    >>> np.quantile(ar, q=0.5)
    2.5
    >>> np.quantile(ar, q=0.5, axis=0, weights=[1, 1])
    array([ 1.5,  2.5,  3.5])
    >>> np.quantile(ar, q=0.5, axis=1, weights=[1, 1, 1])
    array([ 1.,  4.])
    >>> np.quantile(ar, q=0.5, axis=1, weights=[0, 1, 2])
    array([ 2.,  5.])
    >>> np.quantile(ar, q=[0.25, 0.5, 0.75], axis=1, weights=[0, 1, 2])
    array([[ 1.5,  4.5],
           [ 2. ,  5. ],
           [ 2. ,  5. ]])
    """
    from numpy.lib.function_base import (
        np, array, asanyarray, _ureduce,
    )

    q = array(q, dtype=np.float64, copy=True)

    if weights is None:
        wgt = None
    else:
        a = asanyarray(a)
        wgt = np.asanyarray(weights)

        if issubclass(a.dtype.type, (np.integer, bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        broadcastable = False
        if a.ndim == wgt.ndim and a.shape != wgt.shape:
            broadcastable = all([a_dim == w_dim or w_dim == 1
                                 for a_dim, w_dim
                                 in zip(a.shape, wgt.shape)])

        if a.shape != wgt.shape and not broadcastable:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ and not broadcastable.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ "
                    " and not broadcastable.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")
            if not np.issubdtype(wgt.dtype, np.number):
                raise ValueError("All weight entries must be numeric.")

            nan_ws = wgt[np.isnan(wgt)]
            if nan_ws.size > 0:
                raise ValueError("No weight can be NaN.")

            negative_ws = wgt[wgt < 0]
            if negative_ws.size > 0:
                raise ValueError("Negative weight not allowed.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        else:  # same shape, or at least broadcastable
            if axis is None:
                axis = tuple(range(a.ndim))

        scl = wgt.sum(axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")
        # Obtain a weights array of the same shape as reduced a
        wgt = np.broadcast_to(wgt, a.shape)
        wgt, _ = _ureduce(wgt, func=lambda x, **kwargs: x, axis=axis)

    r, k = _ureduce(a, func=_quantile, q=q, axis=axis, weights=wgt, out=out,
                    overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims:
        if q.ndim == 0:
            return r.reshape(k)
        else:
            return r.reshape((len(q),) + k)
    else:
        return r


def _quantile(a, q, axis=None, weights=None, out=None,
              overwrite_input=False, interpolation='linear', keepdims=False):
    from numpy.lib.function_base import (
        np, add, array, asarray, concatenate, intp, take, _ureduce,
    )

    a = asarray(a)
    if q.ndim == 0:
        # Do not allow 0-d arrays because following code fails for scalar
        zerod = True
        q = q[None]
    else:
        zerod = False

    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.size < 10:
        for i in range(q.size):
            if q[i] < 0. or q[i] > 1.:
                raise ValueError("Quantiles must be in the range [0,1], "
                                 "or percentiles must be [0,100].")
    else:
        # faster than any()
        if np.count_nonzero(q < 0.) or np.count_nonzero(q > 1.):
            raise ValueError("Quantiles must be in the range [0,1] "
                             "or percentiles must be [0,100].")

    # prepare a for partioning
    if overwrite_input:
        if axis is None:
            ap = a.ravel()
        else:
            ap = a
    else:
        if axis is None:
            ap = a.flatten()
        else:
            ap = a.copy()

    if axis is None:
        axis = 0

    # move axis to -1 for np.vectorize() operations.  Will move back later.
    ap = np.swapaxes(ap, axis, -1)

    if weights is None:
        Nx = ap.shape[-1]
        indices = q * (Nx - 1)
    else:
        # need a copy of weights for later array assignment
        weights = np.swapaxes(weights.astype('f8'), axis, -1)
        weights[weights < 0.] = 0.  # negative weights are treated as 0
        # values with weight=0 are assigned minimum value and later moved left
        abs_min = np.amin(ap)
        ap[weights == 0.] = abs_min - 1.

        def _sort_by_index(vector, vec_indices):
            return vector[vec_indices]
        # this func vectorizes sort along axis
        arraysort = np.vectorize(_sort_by_index, signature='(i),(i)->(i)')

        ind_sorted = np.argsort(ap, axis=-1)  # sort values long axis
        ap_sorted = arraysort(ap, ind_sorted).astype('f8')

        n = np.isnan(ap_sorted[..., -1:])
        if n.ndim > 1:
            n = np.swapaxes(n, axis, -1)

        ws_sorted = arraysort(weights, ind_sorted).astype('f8')
        ws_sorted[np.isnan(ap_sorted)] = 0.  # neglect nans from calculation
        nonzero_w_inds = ws_sorted > 0.

        cum_w = ws_sorted.cumsum(axis=-1)
        cum_w_max = cum_w.max(axis=-1)

        # some manipulation to get lower/upper percentage bounds
        normalized_w_upper = (cum_w.T / cum_w_max.T).T
        prior_cum_w = np.roll(normalized_w_upper, 1, axis=-1)
        prior_cum_w[..., 0] = 0.

        w_slice = ws_sorted  # .copy()
        # in case any input weight is less than 1, we renormalize by min
        if True in (ws_sorted[nonzero_w_inds] < 1.0):
            ws_sorted[nonzero_w_inds] =\
                ws_sorted[nonzero_w_inds] / ws_sorted[nonzero_w_inds].min()

        w_slice[nonzero_w_inds] = ((normalized_w_upper[nonzero_w_inds] -
                                    prior_cum_w[nonzero_w_inds]) /
                                   ws_sorted[nonzero_w_inds])

        w_slice = np.roll(w_slice, -1, axis=-1)
        # now create the lower percentage bound
        normalized_w_lower = np.roll(normalized_w_upper + w_slice, 1, axis=-1)
        normalized_w_lower[..., 0] = 0.0

        # now we substract by left-most w_slice value
        new_w_upper = (normalized_w_upper.T - w_slice[..., 0].T).T
        new_w_upper[new_w_upper < 0.0] = 0.0
        new_w_lower = (normalized_w_lower.T - w_slice[..., 0].T).T
        new_w_lower[new_w_lower < 0.0] = 0.0
        new_w_lower[..., 0] = 0.0

        # renormalize by right-most bound
        normalized_w_upper = (new_w_upper.T / new_w_upper[..., -1].T).T
        normalized_w_lower = (new_w_lower.T / new_w_upper[..., -1].T).T

        # combine and resort
        cum_w_bands = np.concatenate([normalized_w_upper, normalized_w_lower],
                                     axis=-1)
        inds_resort = np.argsort(cum_w_bands, axis=-1)
        cum_w_bands = arraysort(cum_w_bands, inds_resort)

        ap = np.concatenate([ap_sorted, ap_sorted], axis=-1)
        ap = arraysort(ap, inds_resort)

        # interpolate
        Nx = ap.shape[-1]
        indices_hard = np.arange(Nx)
        vec_interp_func = np.vectorize(np.interp, signature='(n),(m),(m)->(n)')
        indices = vec_interp_func(q, cum_w_bands, indices_hard)

    if interpolation == 'lower':
        indices = np.floor(indices).astype(intp)
    elif interpolation == 'higher':
        indices = np.ceil(indices).astype(intp)
    elif interpolation == 'midpoint':
        indices = 0.5 * (np.floor(indices) + np.ceil(indices))
    elif interpolation == 'nearest':
        indices = np.around(indices).astype(intp)
    elif interpolation == 'linear':
        pass  # keep index as fraction and interpolate
    else:
        raise ValueError(
            "interpolation can only be 'linear', 'lower' 'higher', "
            "'midpoint', or 'nearest'")

    inexact = np.issubdtype(a.dtype, np.inexact)

    if indices.dtype == intp:

        if weights is None:
            if inexact:
                indices = concatenate((indices, [-1]))  # to move nan's to end
            ap.partition(indices, axis=-1)
            n = np.isnan(ap[..., -1:])
            if inexact:
                indices = indices[:-1]
            if n.ndim > 1:
                n = np.swapaxes(n, axis, -1)

        r = take(ap, indices, axis=-1)

        if r.ndim > 1:
            r = np.swapaxes(r, axis, -1)  # move the axis back

        r = np.moveaxis(r, axis, 0)

        if zerod:
            r = r.squeeze(0)

        if out is not None:
            r = add(r, 0, out=out)

    else:  # weight the points above and below the indices
        indices_below = np.floor(indices).astype(intp)
        indices_above = indices_below + 1
        indices_above[indices_above > Nx - 1] = Nx - 1

        if weights is None:
            if inexact:
                indices_above = concatenate((indices_above, [-1]))
            ap.partition(concatenate((indices_below, indices_above)), axis=-1)
            n = np.isnan(ap[..., -1:])
            if inexact:
                indices_above = indices_above[:-1]
            if n.ndim > 1:
                n = np.swapaxes(n, axis, -1)

        weights_above = indices - indices_below
        weights_below = 1.0 - weights_above

        def _take1d(vec, inds, wts):
            return take(vec, inds) * wts

        vec_take = np.vectorize(_take1d, signature='(n),(m),(m)->(m)')

        x1 = vec_take(ap, indices_below, weights_below)
        x2 = vec_take(ap, indices_above, weights_above)

        if x1.ndim > 1:  # move the axis back
            x1 = np.swapaxes(x1, axis, -1)
            x2 = np.swapaxes(x2, axis, -1)

        x1 = np.moveaxis(x1, axis, 0)
        x2 = np.moveaxis(x2, axis, 0)

        if zerod:
            x1 = x1.squeeze(0)
            x2 = x2.squeeze(0)

        if out is not None:
            r = add(x1, x2, out=out)
        else:
            r = add(x1, x2)

    if np.any(n):
        warnings.warn("Invalid value encountered in percentile",
                      RuntimeWarning, stacklevel=3)
        if zerod:
            if ap.ndim == 1:
                if out is not None:
                    out[...] = a.dtype.type(np.nan)
                    r = out
                else:
                    r = a.dtype.type(np.nan)
            else:
                r[n.squeeze(axis)] = a.dtype.type(np.nan)
        else:
            if r.ndim == 1:
                r[:] = a.dtype.type(np.nan)
            else:
                r[..., n.squeeze(axis)] = a.dtype.type(np.nan)

    return r


def contour_levels(contour_grid, p_levels):
    import numpy

    if sorted(p_levels) != p_levels:
        raise ValueError(
            "p_levels must be sorted, got: '{p_levels}'"
            .format(p_levels=p_levels)
        )

    sorted_vals = numpy.sort(contour_grid, axis=None)
    n = len(sorted_vals)

    cdf = numpy.cumsum(sorted_vals)
    cdf /= cdf[-1]

    P_prev = -numpy.inf
    i_start = 0
    vals = numpy.empty_like(p_levels)
    for j, p in enumerate(p_levels):
        for i in range(i_start, n):
            P = cdf[i]

            if P_prev <= p <= P:
                # Use linear interpolation to get value
                val = (sorted_vals[i]-sorted_vals[i-1])/(P-P_prev) * p
                i_start = i
                P_prev = P
                break
            else:
                P_prev = P
        else:
            val = numpy.nan

        vals[j] = val

    return vals


class gaussian_kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Taken from <https://gist.github.com/tillahoffmann/f844bce2ec264c1c8cb5>

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, shape (n, ), optional, default: None
        An array of weights, of the same shape as `x`.  Each value in `x`
        only contributes its associated weight towards the bin count
        (instead of 1).

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : float
        Effective sample size using Kish's approximation.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.pdf(points) : ndarray
        Alias for ``kde.evaluate(points)``.
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = np.random.normal(size=n)
    >>>     m2 = np.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """
    def __init__(self, dataset, bw_method=None, bw_coef=1.0, weights=None):
        import numpy as np

        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape

        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n

        # Compute the effective sample size 
        # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        self.neff = 1.0 / np.sum(self.weights ** 2)
        self.bw_coef = bw_coef

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        import numpy as np
        from scipy.spatial.distance import cdist

        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        # compute the normalised residuals
        chi2 = cdist(points.T, self.dataset.T, 'mahalanobis', VI=self.inv_cov) ** 2
        # compute the pdf
        result = np.sum(np.exp(-.5 * chi2) * self.weights, axis=1) / self._norm_factor

        return result

    __call__ = evaluate

    def scotts_factor(self):
        import numpy as np
        return (
            self.bw_coef *
            np.power(self.neff, -1./(self.d+4))
        )

    def silverman_factor(self):
        import numpy as np
        return (
            self.bw_coef *
            np.power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))
        )

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        import numpy as np
        from six import string_types

        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        import numpy as np

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Compute the mean and residuals
            _mean = np.sum(self.weights * self.dataset, axis=1)
            _residual = (self.dataset - _mean[:, None])
            # Compute the biased covariance
            self._data_covariance = np.atleast_2d(np.dot(_residual * self.weights, _residual.T))
            # Correct for bias (http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
            self._data_covariance /= (1 - np.sum(self.weights ** 2))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) #* self.n
