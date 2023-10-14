# [vt.py](vt.py)

This Python module contains several functions and a command-line interface for computing the sensitive time-volume (VT) for gravitational wave sources. The VT represents the effective time and volume over which a gravitational wave detector is sensitive to binary black hole (BBH) mergers. Here's an explanation of each function in a formal standard:

1. `draw_thetas(N)`

   - Description: This function generates `N` random angular factors for the signal-to-noise ratio (SNR) calculations. It generates random angles, `cos_thetas`, `cos_incs`, `phis`, and `zetas`, and then computes the gravitational wave polarization components (`Fps` and `Fxs`) for each angle combination.
   - Input:
     - `N` (int): The number of random angular factors to generate.
   - Output:
     - Returns an array of SNR factors.

2. `next_pow_two(x)`

   - Description: This function returns the next integer power of two above a given number `x`.
   - Input:
     - `x` (float): The number for which the next power of two needs to be found.
   - Output:
     - Returns the next power of two greater than or equal to `x`.

3. `optimal_snr(m1_intrinsic, m2_intrinsic, z, psd_fn=None)`

   - Description: This function calculates the optimal SNR for a gravitational wave signal from a binary black hole merger.
   - Inputs:
     - `m1_intrinsic` (float): Mass of the first black hole in the source frame.
     - `m2_intrinsic` (float): Mass of the second black hole in the source frame.
     - `z` (float): Redshift.
     - `psd_fn` (function, optional): A function that returns the detector's power spectral density (PSD). Default is the early aLIGO high sensitivity PSD.
   - Output:
     - Returns the optimal SNR for a face-on, overhead source.

4. `fraction_above_threshold(m1_intrinsic, m2_intrinsic, z, snr_thresh, psd_fn=None)`

   - Description: This function calculates the fraction of sources above a given SNR threshold.
   - Inputs:
     - `m1_intrinsic` (float): Mass of the first black hole in the source frame.
     - `m2_intrinsic` (float): Mass of the second black hole in the source frame.
     - `z` (float): Redshift.
     - `snr_thresh` (float): SNR threshold.
     - `psd_fn` (function, optional): A function that returns the detector's PSD. Default is the early aLIGO high sensitivity PSD.
   - Output:
     - Returns the fraction of sources that have an SNR above the given threshold.

5. `vt_from_mass(m1, m2, thresh, analysis_time, calfactor=1.0, psd_fn=None)`

   - Description: This function calculates the sensitive time-volume for a given binary black hole system.
   - Inputs:
     - `m1` (float): Mass of the first black hole in the source frame.
     - `m2` (float): Mass of the second black hole in the source frame.
     - `thresh` (float): SNR threshold.
     - `analysis_time` (float): The total detector-frame searched time (in years).
     - `calfactor` (float, optional): A fudge factor applied to the final result. Default is 1.0.
     - `psd_fn` (function, optional): A function that returns the detector's PSD. Default is the early aLIGO high sensitivity PSD.
   - Output:
     - Returns the sensitive time-volume in comoving Gpc^3-yr.

6. `VTFromMassTuple`

   - Description: This class is used to store parameters for VT calculation.
   - Inputs:
     - `thresh` (float): SNR threshold.
     - `analyt` (float): Analysis time (in years).
     - `calfactor` (float): Calibration factor.
     - `psd_fn` (function): Function that returns the detector's PSD.

7. `vts_from_masses(m1s, m2s, thresh, analysis_time, calfactor=1.0, psd_fn=None, processes=None)`

   - Description: This function calculates an array of VTs for multiple binary black hole systems using multiprocessing for efficient computation.
   - Inputs:
     - `m1s` (array): Array of source-frame mass 1 values.
     - `m2s` (array): Array of source-frame mass 2 values.
     - `thresh` (float): SNR threshold.
     - `analysis_time` (float): Analysis time (in years).
     - `calfactor` (float, optional): A fudge factor applied to the final result. Default is 1.0.
     - `psd_fn` (function, optional): A function that returns the detector's PSD. Default is the early aLIGO high sensitivity PSD.
     - `processes` (int, optional): The number of processes to use for parallel computation.
   - Output:
     - Returns an array of VTs corresponding to the given binary black hole systems.

8. `interpolate_hdf5(hdf5_file)`

   - Description: This function wraps the `interpolate` function using data from an HDF5 file. The HDF5 file should contain grids of mass values and corresponding VT values.
   - Input:
     - `hdf5_file`: An HDF5 file containing mass and VT grids.
   - Output:
     - Returns a function for VT interpolation.

9. `interpolate(m1_grid, m2_grid, VT_grid)`

   - Description: This function returns a function for VT interpolation based on provided mass and VT grids.
   - Inputs:
     - `m1_grid` (array): Mass grid for source-frame mass 1.
     - `m2_grid` (array): Mass grid for source-frame mass 2.
     - `VT_grid` (array): Grid of VT values corresponding to mass values.
   - Output:
     - Returns a function that can interpolate VT for given mass values.

10. `_get_args(raw_args)`

    - Description: This function parses command-line arguments for the main program that generates VT data.
    - Input:
      - `raw_args` (list): Raw command-line arguments passed to the script.
    - Output:
      - Returns a parsed argument namespace.

11. `_main(raw_args=None)`

    - Description: This is the main function that generates VT data based on command-line arguments and stores it in an HDF5 file.

12. `_get_args_plot(raw_args)`

    - Description: This function parses command-line arguments for the main program that generates plots based on VT data.
    - Input:
      - `raw_args` (list): Raw command-line arguments passed to the script.
    - Output:
      - Returns a parsed argument namespace.

13. `_main_plot(raw_args=None)`
    - Description: This is the main function that generates plots based on VT data and command-line arguments.

The module also includes a block at the end to handle command-line execution of the main programs. It reads command-line arguments, performs the specified calculations, and may generate VT data or plots depending on the
