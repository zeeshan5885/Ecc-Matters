# Flow of the Code 

As an overview the code works this way:
- Parse command-line arguments using _get_args().
- Initialize random state and set maximum total mass to provided M_max.
- Calculate constants based on fixed values.
- Read data from event files.
- Load VTs from an HDF5 file.
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

   - This function calculates the intensity (probability density) of a binary black hole merger event rate given individual and population parameters.
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
   - It generates initial positions for walkers based on specified constraints.
   - The constraints can be uniform ranges for log rate, alpha, m_min, and m_max.
   - It returns a 2D array of initial walker positions.

6. **\_get_args()**:

   - This function parses command-line arguments using the `argparse` module.
   - It defines various command-line arguments for controlling the MCMC sampler's behavior.
   - The parsed arguments are returned as a namespace.
   

## Detailed Description of _main():

**_main() Function**

The `_main()` function is the central entry point for the code's execution. It orchestrates the entire analysis process, including data processing, MCMC sampling, and result storage. Here's a step-by-step breakdown of what `_main()` does more oriented towards code:

1. **Argument Parsing**: It starts by parsing command-line arguments using the `_get_args()` function. This allows users to provide input parameters and configuration options for the analysis. The parsed arguments are stored in the `cli_args` variable.

2. **Random State Initialization**: After parsing arguments, the code initializes a random state using the `numpy.random.RandomState` function. This random state is used for generating random numbers and ensuring reproducibility in simulations. The seed for the random state can be provided as a command-line argument.

3. **Setting Maximum Total Mass (M_max)**: The code sets the maximum total mass `M_max`. This value is used in various calculations related to binary black hole mergers.

4. **Determining CPU Count**: The number of available CPU cores (number of CPU threads) is determined using the `cpu_count()` function. This information can be used to optimize parallel processing.

5. **Handling Constants**: The code checks for the presence of fixed constants for population parameters, such as `log_rate`, `alpha`, `m_min`, and `m_max`. If these constants are provided as command-line arguments, they are stored in the `constants` dictionary.

6. **Reading Data from Event Files**: The code proceeds to read data from event files. These event files contain information about binary black hole mergers. The data is processed and stored in the `data_posterior_samples` list.

7. **Loading VTs**: The code opens an HDF5 file that contains VTs (values associated with binary black hole masses). It loads these VTs for later use in the Monte Carlo integration.

8. **Determining Number of Dimensions**: The number of dimensions (`ndim`) for the Markov Chain Monte Carlo (MCMC) sampler is determined. This is related to the number of population parameters and constants.

9. **Initializing Walkers**: The code initializes walker positions based on user-defined constraints or uniform random distributions. This step ensures that the MCMC sampler starts from an appropriate state.

10. **Creating HDF5 File for Posterior Samples**: An HDF5 file is created to store the posterior samples generated during the MCMC sampling. This file will contain the results of the Bayesian analysis.

11. **Storing Initial Positions and Constants**: Initial walker positions are stored in the HDF5 file for reference. Additionally, constants and limits related to population parameters are stored as attributes of the HDF5 file.

12. **Performing MCMC Sampling**: The core of the analysis is conducted by calling the `run_mcmc()` function. This function performs the MCMC sampling, where the likelihood, prior probabilities, and other information are used to estimate the posterior distribution of population parameters. A detailed description of `mcmc.run_mcmc()` is present [here](mcmc_explaination.md).

13. **Calculating Intensity and Expected Value**: At different points within the MCMC sampling, functions like `intensity()` and `expval_mc()` are called to calculate the intensity of binary black hole mergers and the expected value of the intensity using Monte Carlo integration.

14. **Interpolating Values and Log Prior Probability**: Functions like `VT_interp()` and `log_prior_pop()` are used to interpolate values based on binary black hole masses and calculate the log prior probability for population parameters.

15. **Pre-computing Quantities**: The code uses functions `before_prior_aux_fn()` and `after_prior_aux_fn()` to pre-compute quantities related to the rate and probability density, which help reduce runtime.

16. **Saving Posterior Samples**: Posterior samples generated during the MCMC sampling are saved to the HDF5 file created earlier. This file will contain the results of the Bayesian analysis.




