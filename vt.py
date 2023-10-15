"""
This module is a slightly modified version of a module written by Will Farr,
available at
<https://git.ligo.org/RatesAndPopulations/O2Populations/blob/fe81c5d064283c94e12a19567e020b9e9930efef/code/vt.py>

The main changes made here are for data I/O, and the majority of the
implementation is the original code written by Will Farr.
"""

from __future__ import print_function

import argparse
import multiprocessing as multi
import operator
import sys

import astropy.cosmology as cosmo
import astropy.units as u
import h5py
import lal
import lalsimulation as ls
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from numpy import array, cos, linspace, sin, square, trapz

from . import gw

# from pylab import *


def draw_thetas(N):
    """Draw `N` random angular factors for the SNR.

    Theta is as defined in [Finn & Chernoff
    (1993)](https://ui.adsabs.harvard.edu/#abs/1993PhRvD..47.2198F/abstract).
    """

    cos_thetas = np.random.uniform(low=-1, high=1, size=N)
    cos_incs = np.random.uniform(low=-1, high=1, size=N)
    phis = np.random.uniform(low=0, high=2 * np.pi, size=N)
    zetas = np.random.uniform(low=0, high=2 * np.pi, size=N)

    Fps = 0.5 * cos(2 * zetas) * (1 + square(cos_thetas)) * cos(2 * phis) - sin(2 * zetas) * cos_thetas * sin(2 * phis)
    Fxs = 0.5 * sin(2 * zetas) * (1 + square(cos_thetas)) * cos(2 * phis) + cos(2 * zetas) * cos_thetas * sin(2 * phis)

    return np.sqrt(0.25 * square(Fps) * square(1 + square(cos_incs)) + square(Fxs) * square(cos_incs))


def next_pow_two(x):
    """Return the next (integer) power of two above `x`."""

    x2 = 1
    while x2 < x:
        x2 = x2 << 1
    return x2


def optimal_snr(m1_intrinsic, m2_intrinsic, z, psd_fn=None):
    """Return the optimal SNR of a signal.

    :param m1_intrinsic: The source-frame mass 1.

    :param m2_intrinsic: The source-frame mass 2.

    :param z: The redshift.

    :param psd_fn: A function that returns the detector PSD at a given
      frequency (default is early aLIGO high sensitivity, defined in
      [P1200087](https://dcc.ligo.org/LIGO-P1200087/public).

    :return: The SNR of a face-on, overhead source.

    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087

    # Get dL, Gpc
    dL = cosmo.Planck15.luminosity_distance(z).to(u.Gpc).value

    # Basic setup: min frequency for w.f., PSD start freq, etc.
    fmin = 19.0
    fref = 40.0
    psdstart = 20.0

    # This is a conservative estimate of the chirp time + MR time (2 seconds)
    tmax = (ls.SimInspiralChirpTimeBound(
        fmin,
        m1_intrinsic * (1 + z) * lal.MSUN_SI,
        m2_intrinsic * (1 + z) * lal.MSUN_SI,
        0.0,
        0.0,
    ) + 2)

    df = 1.0 / next_pow_two(tmax)
    fmax = 2048.0  # Hz --- based on max freq of 5-5 inspiral

    # Generate the waveform, redshifted as we would see it in the detector, but with zero angles (i.e. phase = 0, inclination = 0)
    ## Will's version -- apparently from a different version of lalsim
    ##    hp, hc = ls.SimInspiralChooseFDWaveform((1+z)*m1_intrinsic*lal.MSUN_SI, (1+z)*m2_intrinsic*lal.MSUN_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dL*1e9*lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0, df, fmin, fmax, fref, None, ls.IMRPhenomPv2)
    hp, hc = ls.SimInspiralChooseFDWaveform(
        (1 + z) * m1_intrinsic * lal.MSUN_SI,
        (1 + z) * m2_intrinsic * lal.MSUN_SI,
        0,
        0,
        0,
        0,
        0,
        0,
        dL * 1e9 * lal.PC_SI,
        0,
        0,
        0,
        0,
        0,
        df,
        fmin,
        fmax,
        fref,
        None,
        ls.IMRPhenomPv2,
    )

    Nf = int(round(fmax / df)) + 1
    fs = linspace(0, fmax, Nf)
    sel = fs > psdstart

    # PSD
    sffs = lal.CreateREAL8FrequencySeries("psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0])
    psd_fn(sffs, psdstart)

    return ls.MeasureSNRFD(hp, sffs, psdstart, -1.0)


# Variable only used by ``fraction_above_threshold``. The value is constant, and
# needs to only be computed once. It also takes a non-negligible amount of time
# to compute, and so it is computed the first time ``fraction_above_threshold``
# is called, as one might import this module without calling that function.
# NOTE: This is a change from Will Farr's original code.
_thetas = None


def fraction_above_threshold(m1_intrinsic, m2_intrinsic, z, snr_thresh, psd_fn=None):
    """Returns the fraction of sources above a given threshold.

    :param m1_intrinsic: Source-frame mass 1.

    :param m2_intrinsic: Source-frame mass 2.

    :param z: Redshift.

    :param snr_thresh: SNR threshold.

    :param psd_fn: Function computing the PSD (see :func:`optimal_snr`).

    :return: The fraction of sources that are above the given
      threshold.

    """
    global _thetas

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087

    # Compute ``_thetas`` once and for all. It is stored as a global variable,
    # but it should also not be used outside of this function.
    if _thetas is None:
        _thetas = draw_thetas(10000)

    if z == 0.0:
        return 1.0

    rho_max = optimal_snr(m1_intrinsic, m2_intrinsic, z, psd_fn=psd_fn)

    # From Finn & Chernoff, we have SNR ~ theta*integrand, assuming that the polarisations are
    # orthogonal
    theta_min = snr_thresh / rho_max

    if theta_min > 1:
        return 0.0
    else:
        return np.mean(_thetas > theta_min)


def vt_from_mass(m1, m2, thresh, analysis_time, calfactor=1.0, psd_fn=None):
    """Returns the sensitive time-volume for a given system.

    :param m1: Source-frame mass 1.

    :param m2: Source-frame mass 2.

    :param analysis_time: The total detector-frame searched time.

    :param calfactor: Fudge factor applied multiplicatively to the final result.

    :param psd_fn: Function giving the assumed single-detector PSD
      (see :func:`optimal_snr`).

    :return: The sensitive time-volume in comoving Gpc^3-yr (assuming
      analysis_time is given in years).

    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087

    def integrand(z):
        if z == 0.0:
            return 0.0
        else:
            return (4 * np.pi * cosmo.Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value / (1 + z) *
                    fraction_above_threshold(m1, m2, z, thresh))

    zmax = 1.0
    zmin = 0.001
    assert fraction_above_threshold(m1, m2, zmax, thresh) == 0.0
    assert fraction_above_threshold(m1, m2, zmin, thresh) > 0.0
    while zmax - zmin > 1e-3:
        zhalf = 0.5 * (zmax + zmin)
        fhalf = fraction_above_threshold(m1, m2, zhalf, thresh)

        if fhalf > 0.0:
            zmin = zhalf
        else:
            zmax = zhalf

    zs = linspace(0.0, zmax, 20)
    ys = array([integrand(z) for z in zs])
    return calfactor * analysis_time * trapz(ys, zs)


class VTFromMassTuple(object):

    def __init__(self, thresh, analyt, calfactor, psd_fn):
        self.thresh = thresh
        self.analyt = analyt
        self.calfactor = calfactor
        self.psd_fn = psd_fn

    def __call__(self, m1m2):
        m1, m2 = m1m2
        return vt_from_mass(m1, m2, self.thresh, self.analyt, self.calfactor, self.psd_fn)


def vts_from_masses(
    m1s,
    m2s,
    thresh,
    analysis_time,
    calfactor=1.0,
    psd_fn=None,
    processes=None,
):
    """Returns array of VTs corresponding to the given systems.

    Uses multiprocessing for more efficient computation.
    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDaLIGOEarlyHighSensitivityP1200087

    vt_m_tuple = VTFromMassTuple(thresh, analysis_time, calfactor, psd_fn)

    pool = multi.Pool(processes=processes)
    try:
        vts = array(pool.map(vt_m_tuple, zip(m1s, m2s)))
    finally:
        pool.close()

    return vts


def interpolate_hdf5(hdf5_file):
    """
    A convenience function which wraps :py:func:`interpolate`, but given an HDF5
    file. The HDF5 file should contain (at least) the following three datasets:
    (``m1``, ``m2``, ``VT``), which should be arrays appropriate to pass as the
    (``m1_grid``, ``m2_grid``, ``VT_grid``) arguments to :py:func:`interpolate`.
    """
    m1_grid = hdf5_file["m1"][:]
    m2_grid = hdf5_file["m2"][:]
    VT_grid = hdf5_file["VT"][:]

    return interpolate(m1_grid, m2_grid, VT_grid)


def interpolate(m1_grid, m2_grid, VT_grid):
    """
    Return a function, ``VT(m_1, m_2)``, given its value computed on a grid.
    Uses linear interpolation via ``scipy.interpolate.interp2d`` with
    ``kind="linear"`` option set.

    :param m1_grid: Source-frame mass 1.

    :param m2_grid: Source-frame mass 2.

    :param VT_grid: Sensitive volume-time products corresponding to each m1,m2.

    :return: A function ``VT(m_1, m_2)``.
    """

    #    print(m1_grid,m2_grid)
    points = (m1_grid[0], m2_grid[:, 0])
    #    values = VT_grid.flatten()
    #    print(points)
    interpolator = (
        scipy.interpolate.RegularGridInterpolator(  # scipy.interpolate.interp2d(
            points, VT_grid, method="linear", bounds_error=False, fill_value=0))

    return interpolator


def _get_args(raw_args):
    """
    Parse command line arguments when run in CLI mode.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("m_min", type=float)
    parser.add_argument("m_max", type=float)
    parser.add_argument("n_samples", type=int)
    parser.add_argument("duration", type=float)
    parser.add_argument("output")

    parser.add_argument("--threshold", type=float, default=8.0)
    parser.add_argument("--calfactor", type=float, default=1.0)
    parser.add_argument("--psd-fn", default="SimNoisePSDaLIGOEarlyHighSensitivityP1200087")

    return parser.parse_args(raw_args)


def _main(raw_args=None):
    if raw_args is None:
        raw_args = sys.argv[1:]

    args = _get_args(raw_args)

    # Load PSD function from lalsimulation, raising an exception if it
    # doesn't exist.
    try:
        psd_fn = getattr(ls, args.psd_fn)
    except AttributeError as err:
        err.message = "PSD '{}' not found in lalsimulation.".format(args.psd_fn)
        raise

    duration = args.duration / 365.0

    with h5py.File(args.output, "w-") as f:
        masses = np.linspace(args.m_min, args.m_max, args.n_samples)
        M1, M2 = np.meshgrid(masses, masses)
        m1, m2 = M1.ravel(), M2.ravel()

        vts = vts_from_masses(
            m1,
            m2,
            args.threshold,
            duration,
            calfactor=args.calfactor,
            psd_fn=psd_fn,
        )

        VTs = vts.reshape(M1.shape)

        f.create_dataset("m1", data=M1)
        f.create_dataset("m2", data=M2)
        f.create_dataset("VT", data=VTs)


def _get_args_plot(raw_args):
    parser = argparse.ArgumentParser()

    parser.add_argument("table")
    parser.add_argument("output_plot")

    parser.add_argument("m_min", type=float)
    parser.add_argument("m_max", type=float)

    parser.add_argument("--n-samples", default=100, type=int)

    parser.add_argument(
        "--mpl-backend",
        default="Agg",
        help="Backend to use for matplotlib.",
    )

    return parser.parse_args(raw_args)


def _main_plot(raw_args=None):
    if raw_args is None:
        raw_args = sys.argv[1:]

    args = _get_args_plot(raw_args)

    matplotlib.use(args.mpl_backend)

    M_max = args.m_min + args.m_max

    with h5py.File(args.table, "r") as VTs:
        raw_interpolator = interpolate_hdf5(VTs)

        def VT_interp(m1_m2):
            m1 = m1_m2[:, 0]
            m2 = m1_m2[:, 1]

            return raw_interpolator(m1_m2)

    fig, (ax_mchirp, ax_m1_m2) = plt.subplots(1, 2)

    m_linear = np.linspace(args.m_min, args.m_max, args.n_samples)
    m1_mesh, m2_mesh = np.meshgrid(m_linear, m_linear)
    m1, m2 = m1_mesh.ravel(), m2_mesh.ravel()
    m1_m2 = np.column_stack((m1, m2))

    mchirp = gw.chirp_mass_full(m1, m2)

    VT = VT_interp(m1_m2)

    idx_outside = reduce(
        operator.__or__,
        [
            m1 > args.m_max,
            m1 + m2 > M_max,
            m2 < args.m_min,
            m2 > m1,
        ],
    )
    idx_inside = ~idx_outside

    VT[idx_outside] = 0.0

    ctr = ax_m1_m2.contourf(
        m1_mesh,
        m2_mesh,
        np.log10(VT).reshape(m1_mesh.shape),
        100,
        cmap=matplotlib.cm.viridis,
    )
    fig.colorbar(ctr)

    ax_m1_m2.set_ylim([args.m_min, 0.5 * M_max])
    ax_m1_m2.set_xlabel(r"$m_1$")
    ax_m1_m2.set_ylabel(r"$m_2$")

    ax_mchirp.scatter(
        mchirp[idx_inside],
        VT[idx_inside],
        color="black",
        s=10,
    )

    ax_mchirp.set_xlabel(r"$\mathcal{M}_{\mathrm{c}}$")
    ax_mchirp.set_ylabel(r"$\langle VT \rangle$")

    fig.savefig(args.output_plot)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
