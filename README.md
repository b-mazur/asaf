
# Quick and easy analysis of the data from the flat histogram simulations

***ASAF** (Adsorption Simulation Analysis Facilitator)* is a python library created to facilitate the processing and 
analysis of data from grand canonical transition matrix Monte Carlo adsorption simulations.

### Features:

- calculation of the macrostate probability distribution (MPC) from the transition probabilities
- interpolation of the transition probabilities 
- calculation of the adsorption isotherm from the MPD
- calculation of the free energy from the MPD
- temperature extrapolation of the macrostate probability distribution
- saving the isotherms to an AIF file

### Download

To download simply type in your terminal `pip install git+https://github.com/b-mazur/asaf.git`

### Citing

If you use ASAF in your work, please consider citing the following paper: 

> Efficient Modeling of Water Adsorption in MOFs Using Interpolated Transition Matrix Monte Carlo, B. Mazur, L. Firlej, 
> and B. Kuchta, **2024**, ACS Appl. Mater. Interfaces, DOI: [10.1021/acsami.4c02616](https://doi.org/10.1021/acsami.4c02616).

### FAQ

**In my prob files I see `current_cycle` column instead of `macrostate`.**

That's because you're looking at a prob file for a simulation in a particular N macrostate. To calculate the macrostate 
probability distribution you need at least several simulations in different macrostates. The diagram below explains the 
scheme for creating the prob file used by ASAF. 

<img src="https://github.com/b-mazur/asaf/blob/main/docs/prob_files_workflow.png" width="1000" class="center">

**I don't have `.metadata.json` file.**

Currently, the only options are to create this file manually (for example, by copying one of the 
[example files](https://github.com/b-mazur/asaf/blob/main/example/data/prob_MOF-303-E4D4_3.2.3_298.000000_1700.metadata.json) 
and modifying it) or to add a function to your workflow that generates such a file based on RASPA input files. In the future, 
ASAF will be able to do this. 

**Prob files are not generated.**

Make sure that you are using modified version of RASPA.

**My prob files are huge**

In general you can play with `PrintGhostProbabilitesEvery` parameter. A more frequent print will be useful when you want 
to use energy fluctuations to extrapolate MPD, otherwise a value similar to that used in `PrintEvery` will be sufficient 
to monitor the simulation. 
