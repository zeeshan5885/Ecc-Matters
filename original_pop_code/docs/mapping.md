# [Power Law](prob.py)

Power law is defined as,

$$f(x)\propto x^{-\alpha}$$

where $\alpha$ is the power law index and $f:[x_\text{min},x_\text{max}]\rightarrow [0,1]$. By using inverse tranform sampling we can generate random samples from power law distribution. It is implemented as following `powerlaw_rvs`.

# [Joint probability](prob.py)

The equation 7 in the paper i.e.

$$p(m_1,m_2)=\left(\frac{m_2}{m_1}\right)^{k_m}\frac{m_1^{-\alpha}}{m_1-m_\text{min}}\times C(\alpha_m, k_m,m_\text{min},m_\text{max},M_\text{max})$$

is implemented as `joint_pdf`. To calculate $C$ the function is `pdf_const`.

# [Marginal probability](prob.py)

`marginal_rvs` samples the $m_1$ from the joint distribution, by sampling both $m_1$ and $m_2$ from the joint distribution and discarding the $m_2$ samples. `marginal_pdf` calculates the marginal probability of $m_1$, and uses `pdf_const` to calculate the constant $C$.

# [Intensity](original_powerlaw_mcmc.py)

The intensity is defined as,

$$I(m_1,m_2)=\mathcal{R}\times p(m_1,m_2)$$

where $\mathcal{R}$ is the rate of the event. `intensity` calculates the intensity of the event.

# [Log of population prioir](original_powerlaw_mcmc.py)

`log_prior_pop` calculates $\log{p(\Lambda)}$, which is later passed to `emcee` sampler.

# [Initial walker positions](original_powerlaw_mcmc.py)

`init_uniform` generates the initial walker positions for the `emcee` sampler. This can be interpreted as the initial guess for the sampler.
