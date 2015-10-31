### Folders and Files

* Source Code
  * plotter.py  # used indirectly for plotting pie chart
  * dataset.py  # generate dataset object
  * ml.py       # entry driver for experiments

* Input File
  * data.csv    # original dataset in csv form

* Output Files
  * data.pickle # *generated* pickled data
  * models/     # serizalized models for each classifier with a specific parameter
  * results/    # raw experiments data
  * figures/    # generated figures

### Experiment Setup

* Dependency
  * Python2.7.9: https://www.python.org/download/releases/2.7/
  * scikit-learn 0.16: http://scikit-learn.org/stable/install.html

* How to run
  * python ml.py  # in any terminals on Windows/Linux/Mac
