# RMWSPy-Fork
This is a forked verison of RMWSPy. Here we show how to estimate the spatial model using path averaged data. ./rmwspy/gcopula_sparaest_block.py (a modified version of existing ./rmwspy/gcopula_sparaest.py) implements the deconvolution algorithm and the optimization. 1_fully_synthetic.ipynb and 2_radar_and_cml_interpolation.ipynb shows how to run the code and generate the plots in the paper. 

# RMWSPy
Version 1.1

# Introduction
RMWSPy is a Python package for conditional spatial random field simulation and inverse modelling, that provides great flexibility in its range of conditioning constraints and its applications. To get an idea of the underlying RMWS algorithm, the interested reader is referred to https://doi.org/10.1016/j.advwatres.2018.11.012 (S. Hörning and J. Sreekanth and A. Bárdossy, 2019, Computational efficient inverse groundwater modeling using Random Mixing and Whittaker–Shannon interpolation, Advances in Water Resources 123, pages 109-119). A paper describing the actual Python package is now available: S. Hörning, B. Haese, RMWSPy (v 1.1): A Python code for spatial simulation and inversion for environmental applications,
Environmental Modelling & Software, Volume 138, 2021, https://doi.org/10.1016/j.envsoft.2021.104970.

# Contributing
Bug reports, code contributions, and improvements are welcome from the community.

# Examples
In Version 1.1, three examples for RMWSPy are provided. The basic use of the package is described in the basic_example.ipynb Jupyter notebook. 

The second example (cml_example) combines rain gauge observations and Commercial Microwave Link data in an inversion framework to improve spatial precipitation estimation (for more details see  Haese, B., Hörning, S., Chwala, C., Bárdossy, A., Schalge, B., & Kunstmann, H., 2017, Stochastic reconstruction and interpolation of precipitation fields using combined information of commercial microwave links and rain gauges. Water Resources Research, 53, 10,740– 10,756. https://doi.org/10.1002/2017WR021015)

The third example (modflow_example) shows the application of RMWSPy to inverse groundwater modelling. It is a small synthetic problem which uses FloPy (Bakker, M., Post, V., Langevin, C. D., Hughes, J. D.,	White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, https://doi.org/10.1111/gwat.12413) for MODFLOW (Langevin, C.D., Hughes, J.D., Banta, E.R., Provost, A.M., Niswonger, R.G., and Panday, Sorab, 2019, MODFLOW 6 Modular Hydrologic Model version 6.1.0: U.S. Geological Survey Software Release, 12 December 2019, https://doi.org/10.5066/F76Q1VQV) model development in Python.
