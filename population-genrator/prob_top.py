from __future__ import division, print_function


def sample_with_cond(func, shape=(1,), cond=None):
    """
    Returns an array of shape ``shape`` containing samples from a
    non-deterministic function ``func``, which must optionally satisfy some
    condition ``cond``. If a condition is given, ``func`` is called repeatedly
    to replace those samples which do not satisfy ``cond``. ``func`` must be a
    function which takes a valid numpy array shape specifier
    (int or tuple of ints), and returns an array of that shape.

    WARNING: If ``func`` never returns any samples that satisfy ``cond``, this
    function will never terminate.
    """
    # Simple case: no condition given.
    if cond is None:
        return func(shape)

    import numpy

    # Generate initial samples
    samples = numpy.asanyarray(func(shape))

    # Create index array of samples which fail to meet the condition,
    # and count the number of new samples that will be required.
    bad = ~cond(samples)
    N = numpy.count_nonzero(bad)

    # Iteratively replace the bad samples until all samples meet the condition.
    while N:
        # Overwrite bad samples with new (possibly bad) samples
        samples[bad] = func(N)

        # Update index array of bad samples. Note that ``cond`` is only called
        # on the known bad samples, in case the function call is expensive.
        # Note that evaluating ``samples[bad]`` creates a new array, not a
        # view, and may waste a lot of memory.
        new_bad = ~cond(samples[bad])
        bad[bad] = new_bad

        # Count the number of new samples still needed.
        # If zero, the loop terminates.
        N = numpy.count_nonzero(new_bad)

    return samples


def oversample_with_cond(func, size=1, cond=None, oversampling=2):
    assert oversampling >= 1

    # Simple case: no condition given.
    if cond is None:
        return func(size)

    import numpy

    # Generate initial samples
    samples = numpy.asanyarray(func(int(size*oversampling)))

    # Pick out the samples that satisfy the condition,
    # and count the number of good samples for indexing purposes.
    good_samples = samples[cond(samples)]
    N_good = len(good_samples)

    # If we drew enough good samples, return the desired number.
    # Otherwise, add the good samples to the beginning of ``ret``,
    # and continue drawing more samples.
    if N_good >= size:
        return good_samples[:size]
    else:
        # Initialize array with samples to be returned.
        ret = numpy.empty(size, dtype=samples.dtype)
        ret[:N_good] = good_samples

    # Calculate the index to start adding samples to, and the number of samples
    # that are still needed.
    idx_start = N_good
    N_bad = size - idx_start

    while N_bad:
        # Generate more samples
        samples = numpy.asanyarray(func(int(size*oversampling)))

        # Pick out the samples that satisfy the condition,
        # and count the number of good samples for indexing purposes.
        good_samples = samples[cond(samples)]
        N_good = len(good_samples)

        # If we drew enough samples, add just enough to the end of ``ret``
        # and break. Otherwise, add them to the end of ``ret`` and continue.
        if N_good >= N_bad:
            ret[idx_start:idx_start+N_bad] = good_samples[:N_bad]
            break
        else:
            ret[idx_start:idx_start+N_good] = good_samples

            # Calculate the index to start adding samples to, and the number of
            # samples that are still needed.
            idx_start += N_good
            N_bad = size - idx_start

    return ret
