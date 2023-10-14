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

1.  **Integrating the function in a hyperrecatangular region `integrate_box`:**

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

# [mcmc.py](mcmc.py)

# [original_powerlaw_mcmc.py](original_powerlaw_mcmc.py)

# [prob_top.py](prob_top.py)

# [prob.py](prob.py)

The break down of the code into its key components with proper mathematical explanations is,

1. **Power Law Mass Distribution Sampling (powerlaw_rvs):**

   - The `powerlaw_rvs` function generates random samples from a power-law distribution with a specified alpha parameter. The power law is defined as $p(x) \propto x^{-\alpha}$, where $\alpha$ is the power-law index.
   - It uses [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) to draw samples from the distribution.
   - The random samples are generated within the range $[x_{\text{min}}, x_{\text{max}}]$.

2. **Joint Mass Distribution Sampling (joint_rvs):**

   - The `joint_rvs` function draws samples from a joint mass distribution $p(m_1, m_2)$.
   - It first draws samples from the power law mass distribution for $m_1$ using the `powerlaw_rvs` function.
   - Then, it draws samples for $m_2$ from a uniform distribution between $m_{\text{min}}$ and $m_1$.
   - Finally, it rejects samples that do not satisfy the constraint $m_1 + m_2 \le M_{\text{max}}$.

3. **Marginal Mass Distribution Sampling (marginal_rvs):**

   - The `marginal_rvs` function draws samples from the marginal mass distribution $p(m_1)$.
   - It relies on the `joint_rvs` function and discards the $m_2$ component to obtain samples for $m_1$.

4. **Joint PDF Calculation (joint_pdf):**

   - The `joint_pdf` function computes the probability density function (PDF) for the joint mass distribution $p(m_1, m_2)$.
   - The PDF is defined as
     $$p(m_1, m_2) = C(\alpha, m_{\text{min}}, m_{\text{max}}, M_{\text{max}}) \times \frac{m_1^{-\alpha}}{m_1 - m_{\text{min}}}$$
   - The normalization constant is calculated using the `pdf_const` function if not provided explicitly.

5. **Marginal PDF Calculation (marginal_pdf):**

   - The `marginal_pdf` function computes the PDF for the marginal mass distribution $p(m_1)$.
   - It relies on the `joint_pdf` function to obtain the PDF for $m_1$ and discards the $m_2$ component.
   - The normalization constant is calculated using the `pdf_const` function if not provided explicitly.

6. **Normalization Constant Calculation (pdf_const):**

   - The `pdf_const` function computes the normalization constant $C(\alpha, m_{\text{min}}, m_{\text{max}}, M_{\text{max}})$ for the power-law distribution.
   - It handles both special and non-special cases, where the special case corresponds to $\beta = 0$ ($\alpha = 1$).
   - For the special case, it calculates $C$ using specific equations.
   - For non-special cases, it calculates $C$ for non-integer $\beta$ values.
   - It handles cases with and without the $M_{\text{max}}$ cutoff.

7. **Upper Mass Credible Region Calculation (upper_mass_credible_region):**

   - The `upper_mass_credible_region` function calculates the upper mass credible region for a given quantile.
   - It computes the credible region for the marginal mass distribution $p(m_1)$ using a specified quantile.
   - It uses numerical integration to calculate the cumulative distribution function (CDF) and finds the m1 value corresponding to the given quantile.

8. **Upper Mass Credible Region Calculation with Detection Weighting (upper_mass_credible_region_detection_weighted):**
   - This function calculates the upper mass credible region for detection-weighted events.
   - It incorporates detection weighting using the `VT_from_m1_m2` function and follows a similar approach to the `upper_mass_credible_region` function, considering quantile, $\alpha$, $m_{\text{min}}$, $m_{\text{max}}$, and $M_{\text{max}}$.

The code provides tools for generating samples from power-law distributions and calculating PDFs and credible regions, particularly useful in astrophysical contexts where power-law distributions are common models. The mathematical explanations provided above should give you a deeper understanding of how the code works.

# [scaled_events](scaled_events)

# [utils_top.py](utils_top.py)

# [utils.py](utils.py)

# [vt.py](vt.py)
