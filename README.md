# Beetracking
#### Computer Vision Systems Programming, TU Vienna, January 2017

The shared sourcecode was created during a the Computer Vision Systems Programming course at Vienna Technical University. 
Only the generated sourcecodes and report are part of this repository, original and generated datasets are not published. 

### Introduction

### Dataset
The resulting algorithm for detecting and tracking was used to generate a novel dataset from the existing videos.
The new dataset contains two classes of images.

Class | Description | #
--- | --- | ---
| Class1 - positive samples  | Images of bees  | ~30.000 images |
| Class2 - negative samples  | Images of backg | ~20.000 images |

### Run the code
###### Dependencies
To run the code Python 3.5 64bit and Opencv. I ran the code on Microsoft Windows, so all directions I am giving below are also for Windows. 

Needed tools and libraries:
* [Python 3.5 64bit](https://www.python.org/downloads/release/python-353/ "Python 3.5 64bit")
* [Opencv 3.1 for Python 3.5 64bit](http://www.lfd.uci.edu/~gohlke/pythonlibs/f9r7rmd8/opencv_python-3.2.0-cp35-cp35m-win_amd64.whl "Unofficial Python Builds")
* Numpy, pulled from the python repositories via pip
* Imutils, it is used for the Non-Maxima Suppression

If you are using a different version of Python please check out the correct dependencies here: [Unofficial Python Library Builds](http://www.lfd.uci.edu/~gohlke/pythonlibs/ "Unofficial Python Builds")

1. After downloading follow the installer for Python 3.5. 
2. Then install the dependencies by running the following command:
```
pip install numpy imutils
pip install opencv_python‑3.2.0‑cp35‑cp35m‑win_amd64.whl

```
