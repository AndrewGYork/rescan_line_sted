# rescan_line_sted
Scripts used to generate the data in the rescan line-sted manuscript. One script for each data figure: `line_sted_figure_1.py`, `line_sted_figure_2.py`, `line_sted_figure_3.py`. Figure 4 is an illustration, so it has no associated script.

If you want to run the code, you'll need a python 3 environment, and also the python subpackages [Numpy](http://www.numpy.org/), [Scipy](https://www.scipy.org/), and [Matplotlib](http://matplotlib.org/). [Anaconda](https://www.continuum.io/downloads) satisfies these requirements with a single installation.

I'm assuming that you put all the scripts in this repository in the same folder on your local machine. `np_tif.py` and `tqdm.py` are utility scripts that the other scripts depend on (to load TIF files and show nice progress bars, respectively). `test_object_1.tif` and `test_object_2.tif` are images of resolution targets used by the figure-generating scripts.
