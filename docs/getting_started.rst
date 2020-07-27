Getting Started
===============

turboPy Conda environment
-------------------------

-   Create a conda environment for turboPy: ``conda env create -f environment.yml``
-   Activate: ``conda activate turbopy``
-   Install turboPy into the environment (from the main folder where ``setup.py`` is): 
	- Install turboPy in editable mode (i.e. setuptools "develop mode") if you are modifying turboPy itself: ``pip install -e .``
	- If you just plan to develop a code using the existing turboPy framework: ``pip install .``
-   Run tests: ``pytest``


turboPy development environment
-------------------------------

If using ``pylint`` (which you should!) add ``variable-rgx=[a-z0-9_]{1,30}$`` to your ``.pylintrc`` file to allow single character variable names.

Merge requests are encouraged!

Example turboPy app
-------------------

Once you have the turboPy conda environment set up, you can go ahead and write a "turboPy app". The simplest way to get started with writing an app might be to clone an existing example app. 

`This example app <https://github.com/NRL-Plasma-Physics-Division/particle-in-field>`_ computes the motion of a charged particle in an electric field.
