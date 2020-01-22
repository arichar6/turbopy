turboPy
=======================

A lightweight computational physics framework, based on the organization of turboWAVE. Implements a "Simulation, PhysicsModule, ComputeTool" class hierarchy.


To Do
-----

-   Finish implementing the class structures, based on turboWAVE

Merge requests are encouraged!


More Resources
--------------

-   [Official turboWAVE Repo](https://github.com/USNavalResearchLaboratory/turboWAVE)
-   [TurboWAVE Documentation](https://turbowave.readthedocs.io)


turboPy Conda environment
-------------------------

-   Create a conda environment for turboPy: `conda env create -f environment.yml`
-   Activate: `conda activate turbopy`
-   Install turboPy into the environment (from the main folder where setup.py is): 
	- Install turboPy in editable mode (i.e. setuptools "develop mode") if you are 
modifying turboPy itself: `pip install -e .` 
	- If you just plan to develop a code using the existing turboPy framework: `pip install .` 


turboPy development environment
-------------------------------

If using `pylint` (which you should!) add `variable-rgx=[a-z0-9_]{1,30}$` to your .pylintrc file
to allow single character variable names.
