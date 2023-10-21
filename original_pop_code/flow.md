# Flow of the Code 

As an overview the code works this way:
- Parse command-line arguments using _get_args().
- Initialize random state and set maximum total mass to provided M_max.
- Calculate constants based on fixed values.
- Read data from event files.
- Load VTs from an HDF5 file.
- Compute prior probabilities for population parameters.
- Initialize walker positions using init_uniform().
- Create an HDF5 file to store posterior samples.
- Store initial positions in the HDF5 file.
- Create arrays for storing walker positions and log probabilities.
- Store constants and limits in the HDF5 file.
- Perform MCMC sampling using run_mcmc().
- Calculate the intensity of binary black hole mergers using intensity().
- Calculate the expected value of intensity using Monte Carlo integration in expval_mc().
- Interpolate values based on binary black hole masses using VT_interp().
- Calculate the logarithm of the prior probability for population parameters using log_prior_pop().
- Pre-compute quantities related to the rate and probability density before and after the evaluation of population parameter prior probabilities using before_prior_aux_fn() and after_prior_aux_fn().
- Save posterior samples to the HDF5 file.

------- 
The helper functions referenced above are listed below with description of each of them:

1. **intensity()**:

   - This function calculates the intensity (probability density) of a binary black hole merger event given individual and population parameters.
   - The parameters include `indiv_params` (masses of the binary black holes), `pop_params` (population parameters such as the power-law index, mass limits, and rate), and `aux_info` (auxiliary information like precomputed constants).
   - It returns the intensity, which is the rate times the joint probability density of the individual masses and population parameters.

2. **expval_mc()**:

   - This function calculates the expected value of the intensity using Monte Carlo integration.
   - It uses the intensity function, population parameters, and additional arguments to perform the integration.
   - The result is the expected rate of binary black hole mergers.
   - The `err_abs` and `err_rel` parameters control the accuracy of the Monte Carlo integration.

3. **VT_interp()**:

   - This function interpolates a quantity based on the masses of binary black holes.
   - It takes a 2D array of masses `m1_m2` and uses an interpolator (`raw_interpolator`) to estimate a value at those mass combinations.
   - This function is used in the Monte Carlo integration.

4. **log_prior_pop()**:

   - This function calculates the logarithm of the prior probability for the population parameters.
   - It checks whether the population parameters (log rate, alpha, m_min, m_max) fall within specified limits and scales.
   - If the parameters are outside the allowed ranges, it returns negative infinity (indicating an impossible prior).
   - This function is used to define the prior for the population parameters.

5. **init_uniform()**:

   - This function initializes walker positions for the MCMC (Markov Chain Monte Carlo) sampler.
   - It generates initial positions for walkers based on provided or specified constraints.
   - The constraints can be uniform ranges for log rate, alpha, m_min, and m_max.
   - It returns a 2D array of initial walker positions.

6. **\_get_args()**:

   - This function parses command-line arguments using the `argparse` module.
   - It defines various command-line arguments for controlling the MCMC sampler's behavior.
   - The parsed arguments are returned as a namespace.
   

TODO: Give good desc of _main(), I am not satisfied yet.
