# Beetracking
#### Computer Vision Systems Programming, TU Vienna, January 2017

<a href="https://www.youtube.com/watch?v=2fiYIid4Sl4
" target="_blank"><img src="http://img.youtube.com/vi/2fiYIid4Sl4/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="420" border="10" /></a>

This work was created within the course Computer Vision Systems Programming at Vienna University of Technology. Only the generated sourcecode and report are part of this repository, original and generated datasets are not published. This document is an overview over the accomplished results, the full report can be found here: [Report](https://github.com/dschoerk/beetracking/blob/master/docs/beetracking_finalreport.pdf)

Three main tasks were solved in this project:

1. Per frame detection and continuous tracking of bees [Source](BeeTracker.py)
2. Generation of a bee dataset for machine learning methods [Source](BeeTracker.py#L147-L154) [Source](BeeTracker.py#L159-L233)
3. Sliding window based bee detection based on a Support Vector Machine (experimental) [Source](TrainSvm.py)

##### Running the code
The code was developed on Microsoft Windows 10, naturally the guidlines below only apply to Windows users.

Needed tools and libraries:
* [Python 3.5 64bit](https://www.python.org/downloads/release/python-353/ "Python 3.5 64bit")
* [Opencv 3.1 for Python 3.5 64bit](http://www.lfd.uci.edu/~gohlke/pythonlibs/f9r7rmd8/opencv_python-3.2.0-cp35-cp35m-win_amd64.whl "Unofficial Python Builds")
* Numpy
* Imutils, it is used for the Non-Maxima Suppression

> If you are using a different version of Python please check out the correct dependencies here: [Unofficial Python Library Builds](http://www.lfd.uci.edu/~gohlke/pythonlibs/ "Unofficial Python Builds")

1. After downloading follow the installer for Python 3.5. 
2. Then install the dependencies by running the following command:
```
pip install numpy imutils
pip install opencv_python‑3.2.0‑cp35‑cp35m‑win_amd64.whl

```
