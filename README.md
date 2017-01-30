# Beetracking
#### Computer Vision Systems Programming, TU Vienna, January 2017

<a href="http://www.youtube.com/watch?feature=player_embedded&v=BZV3UvgCKPI
" target="_blank"><img src="http://img.youtube.com/vi/BZV3UvgCKPI/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="320" border="10" /></a>

(Old video, to be replaced with a newer version)

This work was created within the course Computer Vision Systems Programming at Vienna University of Technology. 
Only the generated sourcecodes and report are part of this repository, original and generated datasets are not published. 

Three main tasks were solved in this project:

1. Per video frame detection of bees
2. Continuous tracking of individual bees
3. Generation of a bee dataset for machine learning methods (+experimental classification with SVM)

### Introduction
Bee tracking with stereo cameras [[0]](#chiron-et-al)  Controlled environment, introduce the test data and describe the task itself

whats good, whats bad about the given data

### Method

#### Detection
#### Tracking
#### Dataset
More diverse data would be good, samples over all data every 10 seconds. Higher variety in terms of lighting conditions, bees etc.

#### Training and Classification with SVM


Pedestrian classification [[1]](#dalal-et-al)

What did the SVM actually learn? Bees, multiple bees, parts of the lane

### Limitations & Suggested Improvements


### Appendix A : Results
The resulting algorithm for detecting and tracking was used to generate a novel dataset from the existing videos.
The new dataset contains two classes of images.

Class | Description | #
--- | --- | ---
| Class1 - positive samples  | Images of bees  | ~30.000 images |
| Class2 - negative samples  | Images of backg | ~20.000 images |

### Appendix B : Running the code
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

##### Chiron et al. 
Chiron, Guillaume, Petra Gomez-Krämer, and Michel Ménard. "Detecting and tracking honeybees in 3D at the beehive entrance using stereo vision." EURASIP Journal on Image and Video Processing 2013.1 (2013): 1-17

##### Dalal et al.
Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.



