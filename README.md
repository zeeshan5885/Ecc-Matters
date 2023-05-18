
Based on https://git.ligo.org/daniel.wysocki/bayesian-parametric-population-models/-/blob/master/src/pop_models/powerlaw/mcmc.py


Example: might want to look at https://gitlab.com/dwysocki/pop-models-examples

Arguments
* VT: can get from https://gitlab.com/dwysocki/pop-models-examples/-/blob/master/runs/vts/zero-spin-efficient-grid/results/vt_m1_m2_coarse.hdf5
  * or: use 'vt.py' in this directory, to run this code with eccentricity, you need to generate new vt file by running th evt.py
  * problem: vt.py is old, lalsimulation interface ChooseFDWaveform has changed : see https://github.com/oshaughn/research-projects-RIT/blob/master/MonteCarloMarginalizeCode/Code/RIFT/lalsimutils.py#L2918

* problem: old pop_models!
  * solution: check out original pop_models in some ancient virtual environment :  https://gitlab.com/dwysocki/pop-models-examples/-/blob/master/runs/End-of-O2-Mass-and-Spin-Analysis/commit.sha
  * or just use hand-edited code from here

The Whole code is hand edited with ecccentricity and you can directly run it from here. 
