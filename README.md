# Saliency prediction in 360-degree content (images) with traditional "flat" image saliency models via equirectangular format-aware input transformations

Authors:
Mikhail Startsev (mikhail.startsev@tum.de), Michael Dorr (michael.dorr@tum.de).


If you are using this work, please cite the related paper:

    @article{startsev2018aware,
      title = "360-aware saliency estimation with conventional image saliency predictors",
      journal = "Signal Processing: Image Communication",
      volume = "69",
      pages = "43 - 52",
      year = "2018",
      note = "Salient360: Visual attention modeling for 360° Images",
      issn = "0923-5965",
      doi = "https://doi.org/10.1016/j.image.2018.03.013",
      url = "http://www.sciencedirect.com/science/article/pii/S0923596518302595",
      author = "Mikhail Startsev and Michael Dorr",
      keywords = "Saliency prediction, Equirectangular projection, Panoramic images",
    }

This code here utilises existing saliency detection algorithms in traditional, "flat" images. It takes 360-images 
in equirectangular format as input,
applies certain transformation to them, and feeds them into three saliency predictors: GBVS [1], eDN [2], and SAM-ResNet [3].
It then performs inverse transoformations on their outputs, and finally produces equirectangular saliency maps.

With this approach, we participated in the "[Salient360!](https://salient360.ls2n.fr)" Grand Challende at ICME'17. Our algorithm (with combining the saliency 
maps of all three underlying saliency predictors) [has won](https://salient360.ls2n.fr/grand-challenges/icme17/) the "Best Head and Eye Movement Predition" award (i.e. predicting, 
where the eyes of the viewers would land on the equirectangular images, when viewed in a VR headset).

The general pipeline of our approach is outlined in a figure below:

![alt text](https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/overview.png "Overview of our pipeline")


# I. Installation

-------
## Generic
-------

    sudo apt install python-pip
    sudo apt install blender

    sudo pip install h5py
    sudo pip install Pillow


## Installation instructions for each model separately:

----------------
### eDN INSTALLATION
----------------

1. Install dependencies 
  ```
  sudo apt-get install python-matplotlib python-setuptools curl python-dev libxml2-dev libxslt-dev
  ```
  
2. Install liblinear
  
  Download toolbox from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
  or using the command below:

  ```
  wget "http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/liblinear.cgi?+http://www.csie.ntu.edu.tw/~cjlin/liblinear+zip" -O liblinear.zip
  ```
  
  ```
  # extract the zip
  unzip liblinear
  cd liblinear-2.11 
  make
  cd python
  make
  ```
  
3. Install sthor dependencies

  ```
  sudo easy_install pip
  sudo easy_install -U scikit-image
  sudo easy_install -U cython
  sudo easy_install -U numexpr
  sudo easy_install -U scipy
  ```
  
  For speedup, numpy and numexpr should be built against e.g. Intel MKL libraries.
  
4. Install sthor
  
  ```
  git clone https://github.com/nsf-ri-ubicv/sthor.git
  cd sthor/sthor/operation
  ```

In resample_cython_demo.py, line 10: change .lena() call to .ascent() ! Then proceed with the installation.

  ```
  sudo make
  cd ../..
  curl -O http://web.archive.org/web/20140625122200/http://python-distribute.org/distribute_setup.py
  python setup.py install
  ```

#### NB: ADD THE sthor DIRECTORY AND THE liblinear/python DIRECTORY TO YOUR PYTHONPATH

You can add a line 

  	export PYTHONPATH="${PYTHONPATH}:/path/to/sthor/:/path/to/liblinear/python"
    
to your ~/.bashrc and run

> source ~/.bashrc

Note: If you get an error while importing sthor in future, run 

> source ~/.bashrc

again from the working directory.

5. Test sthor installation
  
  ```
  python
  import sthor  # should import without errors
  ```

-----------------
### GBVS INSTALLATION
-----------------

No additional packages required, just having a Matlab installation that can be called from terminal as `matlab`.

-------------------------------------
### Saliency Attentive Model INSTALLATION
-------------------------------------

    sudo pip install keras
    sudo pip install theano tensorflow

    # to make it run on GPU insterad of CPU
    sudo apt install nvidia-cuda-toolkit

Install OpenCV 3.0.0 like here: http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/ , up to step 10
(maybe without verualenv-related instructions)

If some error with memcpy occurs, add this

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
  
to the begining of OpenCV's CMakeLists.txt.


Libgpuarray:

        Follow the instruction here: http://deeplearning.net/software/libgpuarray/installation.html ,
        the sections "Download" and "Step-by-step install: system library (as admin)"
    (use `sudo make install` and `sudo python setup.py install`, if need be).

cuDNN:

    Follow the instructions here: https://askubuntu.com/a/767270


Note: Be sure to have ```"image_dim_ordering": "th"``` and ```"backend": "theano"``` in your keras.json file (normally ~/.keras/keras.json).


If the line 
    from keras import initializations ,
try downgrading Keras and Theano to versions 1.2.2 and 0.9.0, respectively:
    sudo pip install keras==1.2.2
    sudo pip install theano==0.9.0

# II. Test run (on a single image)

To check that everything works, you can use the following commands (`--mode combined` is used to test all steps of the models at once):

    ./360_aware.py /path/to/some/360/image.jpg test_map_eDN.bin --model eDN --mode combined
    ./360_aware.py /path/to/some/360/image.jpg test_map_GBVS.bin --model GBVS --mode combined
    ./360_aware.py /path/to/some/360/image.jpg test_map_SAM.bin --model SAM --mode combined

The basic interface requires 2 positional arguments (input and output files), a `--model` argument, and a `--mode` argument.

# III. Full prediction run

For the arguments to run the models that are described in the paper, see HOW-TO-RUN.txt

You can execute the test code above for images for which the predition is needed, changing the parameters according to the desired model. 
An example bash for-loop is presented below:

> for file in /path/to/test/images/*.jpg ; do im=`basename $file .jpg`; echo $file ; ./360_aware.py $file /path/to/output/folder/"$im".bin --mode combined --model SAM; done ;
