# [mc.py](mc.py)

The `integrate_box` function employs mathematical formulas from Monte Carlo integration to estimate the integral of a given function `f` over a hyperrectangular domain. The key formulas involved are as follows:

1. **Volume of the Hyperrectangular Domain (V):**

   $$V = \prod_i(\text{upper}_i - \text{lower}_i)$$

   This formula calculates the volume of the hyperrectangular domain by taking the product of the differences between the upper and lower bounds ($\text{upper}_i$ and $\text{lower}_i$) along each dimension $i$.

2. **Sample Average (F):**

   $$F = \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{X}_i)$$

   $F$ represents the sample average of the function values $f(\mathbf{X}_i)$ for $N$ random sample points $\mathbf{X}_i$. It is the sum of the function values divided by the total number of samples ($N$). It is also equal to the first moment,

   $$\mu_1 = \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{X}_i)$$

3. **Integral Estimate (I):**

   $$I = V \cdot F$$

   The integral estimate $I$ is calculated by multiplying the volume $V$ of the hyperrectangular domain by the sample average $F$. This provides an estimate of the integral of the function over the specified domain.

4. **Standard Error of the Estimate (dI):**
   $$\mu_2 = \frac{1}{N} \sum_{i=1}^{N} \left(f(\mathbf{X}_i)\right)^2=F2$$
   $$dI = V \cdot \sqrt{\frac{{\mu_2 - \mu^2_1}}{N}}$$

   If the `compute_error` flag is set to `True`, the standard error of the estimate $dI$ is calculated. It represents the uncertainty associated with the integral estimate and is computed using the formula. $F2$ is the sample average of the squared function values.

These formulas are fundamental to the Monte Carlo integration method used in the `integrate_box` function. The integral estimate $I$ represents the expected value of the integral, and the standard error $dI$ provides an estimate of the uncertainty in this value based on the variance of the function values.

1.  **Integrating the function in a hyperrecatangular region (`integrate_box`):**

    - The `integrate_box` function is designed to compute the Monte Carlo integral of a given function $f$ over a hyperrectangular domain defined by lower and upper bounds (`lower` and `upper`) using $N$ samples.

    - It begins by checking and handling the random number generator state (`random_state`) using a utility function utils.`check_random_state`. This ensures that the function can work with different random number generator types and states.

    - The volume of the hyperrectangular domain ($V$) is calculated by taking the product of the differences between the upper and lower bounds along each dimension.
      $$V = \prod_i(\text{upper}_i - \text{lower}_i)$$

    - A key part of Monte Carlo integration is generating random samples within the specified domain. This code uses NumPy's column_stack function to stack arrays of random numbers generated using rng.uniform for each dimension. It creates an array `X` where each row represents a point within the domain.

    - The function `f(X)` is evaluated for each of these sample points, and the results are stored in the array `fX`.

    - The Monte Carlo estimate of the integral is computed as follows:

      - Calculate the sample average of the function values, `F`, which is the sum of `fX` divided by `N`.
        $$F = \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{X}_i)$$
      - Estimate the integral `I` by multiplying the volume `V` by the sample average `F`.
        $$I = V \cdot F$$

    - If the `compute_error` flag is set to `True`, the code proceeds to calculate the standard error of the estimate.

    - To calculate the standard error, compute the sample average of the squared function values, `F2`, which is the sum of the element-wise square of `fX` divided by `N`. Use these values to compute the standard error `dI` as the square root of the variance of the function values, which is `(F2 - F^2) / N`, multiplied by the volume `V`.
      $$dI = V \cdot \sqrt{\frac{{\mu_2 - \mu^2_1}}{N}}$$

    - Depending on whether `compute_error` is `True`, the function returns either just the integral estimate `I` or a tuple containing the integral estimate `I` and the estimated error `dI`.

2.  **Adaptive inregral (`integral_adaptive`):**

    The `integrate_adaptive` function performs adaptive Monte Carlo integration to estimate the integral of the product of two functions: $p(x)$ (a probability distribution) and $f(x)$ (the target function) over some domain. It uses adaptive sampling to achieve accurate results. Below are the mathematical components and explanations for this function:

    - **Adaptive Monte Carlo Integration:**

      - The function aims to compute the integral $\displaystyle\int p(x) \cdot f(x) \, dx$, where $p(x)$ is a probability distribution used for sampling and $f(x)$ is the function to be integrated.

    - **Iterative Sampling:**

      - The method evaluates $f(x)$ at sample points $x$ drawn from $p(x)$.
      - It starts with $2^{10}$ initial iterations and doubles the number of evaluations until convergence or a maximum of $2^{16}$ iterations is reached.

    3.  **Convergence Criteria:**

        - The function allows specifying two optional convergence criteria:
        - `err_abs`: The maximum acceptable standard error for the integral.
        - `err_rel`: The maximum acceptable relative error (standard error divided by the integral).
        - Convergence is determined based on these criteria:
        - If both `err_abs` and `err_rel` are provided, convergence occurs when the current absolute and relative errors are below the specified thresholds.
        - If only one of the criteria is provided, convergence occurs when the corresponding error (absolute or relative) is below the given threshold.
        - If neither criterion is provided, the function does not use convergence as a stopping criterion.

    4.  **Outputs:**

        - The function returns:
        - `integral`: The estimated value of the integral.
        - `err_abs`: The estimated value of the standard error.
        - `err_rel`: The estimated value of the relative error.

    5.  **Iterative Sampling and Convergence Testing:**

        - The function iteratively draws samples from the probability distribution $p(x)$ and evaluates $f(x)$ at these samples.
        - It accumulates the sum of function values in $F$ and the sum of squared function values in $F2$, which are used to calculate the integral and its uncertainty.
        - The number of samples is updated appropriately during iterations.
        - The current estimates of the integral, absolute error, and relative error are calculated.
        - The function tests for convergence based on the specified criteria and terminates if they are met.

    6.  **Diagnostic Plots (Optional):**

        - If the `plot` flag is set to `True`, the function generates diagnostic plots, including integral estimates, absolute errors, and relative errors. These plots can help visualize the convergence process.
        - If `plot` is a filename, the plots are saved to the specified file.

    7.  **Plot Interpretation:**
        - The function shows the plots if not in interactive mode (non-interactive display). Otherwise, it saves the plots to the specified file.
