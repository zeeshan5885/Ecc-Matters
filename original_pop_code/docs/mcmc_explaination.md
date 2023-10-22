# [mcmc.py](mcmc.py)

This code is related to performing a Markov Chain Monte Carlo (MCMC) simulation for a statistical model.

1. **Running the Markov Chain Monte Carlo (`run_mcmc`)**:

   - `intensity_fn`: A function that models the intensity of events, which is often associated with a point process.
   - `expval_fn`: A function that calculates the expected value of a distribution.
   - `data_likelihood_samples`: An array of data samples used for estimating the likelihood of observed data.
   - `log_prior_fn`: A function that calculates the logarithm of the prior probability distribution.
   - `init_state`: Initial states for MCMC walkers.
   - `param_names`: Names of model parameters.
   - `constants`: A dictionary of constant parameters.
   - `data_likelihood_weights`: Weights associated with data samples (optional).
   - `args` and `kwargs`: Additional arguments and keyword arguments for various functions (optional).
   - `before_prior_aux_fn` and `after_prior_aux_fn`: Functions for pre-processing and post-processing auxiliary information (optional).
   - `out_pos` and `out_log_prob`: Arrays for storing MCMC results (optional).
   - `nsamples`: Number of MCMC samples.
   - `rand_state`: Random state for reproducibility (optional).
   - `debug_log_prob`: A flag for debugging (optional).
   - `nthreads`: Number of threads to use (optional).
   - `pool`: A multiprocessing pool for parallel execution (optional).
   - `runtime_sortingfn`: A function for sorting the runtime (optional).
   - `verbose`: A flag for enabling progress updates (optional).
   - `dtype`: Data type for the output arrays (optional).

2. **Log of the posterior (`log_prob`):**

   - `params_free`: Free parameters for which the log-posterior is calculated.
   - `param_names`: Names of model parameters.
   - `constants`: A dictionary of constant parameters.
   - `intensity_fn`: A function that models the intensity of events.
   - `expval_fn`: A function that calculates the expected value.
   - `log_prior_fn`: A function that calculates the logarithm of the prior probability distribution.
   - `data_likelihood_samples`: An array of data samples.
   - `data_likelihood_weights`: Weights associated with data samples (optional).
   - `before_prior_aux_fn` and `after_prior_aux_fn`: Functions for pre-processing and post-processing auxiliary information (optional).
   - `args` and `kwargs`: Additional arguments and keyword arguments for various functions.

3. `get_params` function:
   - `variables`: A list of variable parameters.
   - `constants`: A dictionary of constant parameters.
   - `names`: Names of all model parameters (both variable and constant).

The main goal of this code is to run an MCMC simulation to estimate the posterior distribution of the model parameters given the observed data. The `log_prob` function computes the logarithm of the posterior probability, taking into account the prior distribution and data likelihood. The `run_mcmc` function sets up and executes the MCMC sampling, storing the chain of parameter samples and their log-posterior values.

The code you provided appears to be an implementation of a Markov Chain Monte Carlo (MCMC) sampler, which is a method for performing Bayesian inference. Here's an explanation of the inner workings of the code:

1. **Import Statements**:

   - The code starts with importing necessary modules like `numpy`, `emcee` (a popular MCMC library), and other standard libraries.

2. **Function: `run_mcmc`**:

   - This is the main function for running the MCMC simulation. Let's break down its inner workings:

   - **Initialization**:

     - It sets default values for optional parameters, like `constants`, `data_likelihood_weights`, and others.

   - **Parameter Dimensions**:

     - It calculates the number of free parameters (`ndim`) based on the length of `param_names` minus the number of constants.

   - **Ensemble Parameters**:

     - It determines the number of walkers (`nwalkers`) and the number of individual events (`nindiv`) based on the length of `init_state` and `data_likelihood_samples`.

   - **Data Preprocessing**:

     - It ensures that `data_likelihood_weights` is a list of the same length as `data_likelihood_samples`.

   - **Data Dimensionality Check**:

     - It ensures that all data samples have the same dimensionality.

   - **Initialization of Output Arrays**:

     - It initializes arrays `out_pos` and `out_log_prob` to store the MCMC results. If not provided, it creates empty arrays. If provided, it checks their shape.

   - **MCMC Sampler Creation**:

     - It creates an instance of the `emcee.EnsembleSampler` using the provided or default parameters.

   - **MCMC Sampling**:

     - It iterates through MCMC samples and stores the positions and log-posterior values in `out_pos` and `out_log_prob`, respectively.

   - **Verbose Output**:

     - If `verbose` is set to True, it provides progress updates.

   - **Return Results**:
     - It returns the MCMC chain of parameter samples and their corresponding log-posterior values.

3. **Function: `log_prob`**:

   - This function calculates the logarithm of the posterior probability for a given set of parameters. It combines information from the prior distribution and the likelihood of the observed data. Here's how it works:

   - **Parameter Retrieval**:

     - It retrieves the model parameters (`params`) by combining the free parameters and constants using the `get_params` function.

   - **Prior Calculation**:

     - It calculates the logarithm of the prior probability (`log_pi`) based on the prior distribution and optional preprocessing functions (`before_prior_aux_fn`).

   - **Likelihood Calculation**:

     - It calculates the logarithm of the likelihood contribution from the data samples (`log_events_contribution`). This involves evaluating the intensity of events using `intensity_fn`.

   - **Expected Value Calculation**:

     - It calculates the expected value of the model using `expval_fn`.

   - **Log-Posterior Calculation**:
     - It calculates the log-posterior as the sum of the log-prior
