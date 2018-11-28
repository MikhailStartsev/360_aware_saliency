# Saliency prediction in 360-degree content (images) with traditional "flat" image saliency models via equirectangular format-aware input transformations

Authors:
Mikhail Startsev (mikhail.startsev@tum.de), Michael Dorr (michael.dorr@tum.de).

Project page: http://michaeldorr.de/salient360/


If you are using this work, please cite the related [paper](https://www.sciencedirect.com/science/article/abs/pii/S0923596518302595) (preprint available [here](http://michaeldorr.de/salient360/360_aware_preprint.pdf)):

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
It then performs inverse transformations on their outputs, and finally produces equirectangular saliency maps.

With this approach, we participated in the "[Salient360!](https://salient360.ls2n.fr)" Grand Challende at ICME'17. Our algorithm (with combining the saliency 
maps of all three underlying saliency predictors) [has won](https://salient360.ls2n.fr/grand-challenges/icme17/) the "Best Head and Eye Movement Prediction" award (i.e. predicting, 
where the eyes of the viewers would land on the equirectangular images, when viewed in a VR headset).

The general pipeline of our approach is outlined in a figure below:

![alt text](https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/overview.png "Overview of our pipeline")

The image transformations ("interpretations") we propose are as follows:

* *Continuity-aware:* The image is cut in two halves vertically. The parts are re-stitched in reversed order, resulting in an equirectangular image that is "facing backwards", compared to the original one. After saliency prediction on both of the versions of the scene, the pixel-wise maximum operation is applied. This helps cancel out the border artefacts of the saliency predictors, and results in horizontally-continuous saliency maps. However, vertical scene continuity and image distortions are not addressed.

![alt text](https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/continuity_aware.png "Continuity-aware saliency prediction")


* *Cube map-based*: The equirectangular input is converted to a set of cube faces, which undistorts the image, but looses the context information of the whole scene. We experimented with different methods to regain context (see the paper), but eventually decided for assembling a cutout, and augmenting it with cube faces that would mostly match the borders of the "main" cutout. We call this an *extended cutout*:

![alt text](https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/extended_cutout.png "Continuity-aware saliency prediction")

* *Combined*: This interpretation combines the continuity-aware and the cube map-based interpretations. For the top and the bottom faces of the cube map (the most distorted parts of the equirectangular image), it predicts separate saliency maps. These are then projected back to the equirectangular format (see example below) and combined with the continuity-aware saliency maps via a pixel-wise maximum operation (see pipeline image above). This should both keep the context for the saliency map prediction and address distortions of the input image where those are particularly destructive for the scene content.

![alt text](https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/combined_top_and_bottom.png "Continuity-aware saliency prediction")


We also included an option to add a "centre" (more like "equator") bias to our prediction by adding a (weighted) average saliency map of the training set (see below). This positively affects some of the metrics.

<p align="center">
<img src="https://github.com/MikhailStartsev/360_aware_saliency/blob/master/figures/centre_bias.png" width=512>
</p>

See section **IV** below for instructions to run our best model.

The repository contains the source code of the GBVS, eDN, and SAM models. See `MODIFICATIONS-README.txt` files inside the respective folders for the slight modifications in the processing pipeline that we made (for example, not to re-normalise the saliency maps after prediction, since otherwise combining saliency maps from different interpretation-based images would be done regardless of the scale of the originally predicted saliency values).  

The following folders contain the code from the following repositories:

* `cube2sphere` -- https://github.com/Xyene/cube2sphere
* `edn_cvpr2014` -- https://github.com/coxlab/edn-cvpr2014
* `saliency_attentive_model`  -- https://github.com/marcellacornia/sam

``

# I. Installation

## Generic

    sudo apt install python-pip
    sudo apt install blender

    sudo pip install h5py
    sudo pip install Pillow


## Installation instructions for each model separately:

### eDN INSTALLATION

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

    In resample_cython_demo.py, line 10: change `.lena()` call to `.ascent()`! Then proceed with the installation.

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

    ### GBVS INSTALLATION

    No additional packages required, just having a Matlab installation that can be called from terminal as `matlab`.

    ### Saliency Attentive Model (SAM) INSTALLATION
    ```
    sudo pip install keras
    sudo pip install theano tensorflow

    # to make it run on GPU insterad of CPU
    sudo apt install nvidia-cuda-toolkit
    ```

    Install OpenCV 3.0.0 like here: http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/ , up to step 10 (maybe without verualenv-related instructions).

    If some error with memcpy occurs, add this
        ```
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
        ```
    to the begining of OpenCV's CMakeLists.txt.


    * Libgpuarray:

        Follow the instruction here: http://deeplearning.net/software/libgpuarray/installation.html ,
        the sections "Download" and "Step-by-step install: system library (as admin)"
    (use `sudo make install` and `sudo python setup.py install`, if need be).

    * cuDNN:

        Follow the instructions here: https://askubuntu.com/a/767270


    Note: Be sure to have `"image_dim_ordering": "th"` and `"backend": "theano"` in your keras.json file (normally ~/.keras/keras.json).


    If the line 
        from keras import initializations ,
    try downgrading Keras and Theano to versions 1.2.2 and 0.9.0, respectively:
        sudo pip install keras==1.2.2
        sudo pip install theano==0.9.0

# II. Test run (on a single image)

To check that everything works, you can use the following commands (`--mode combined` is used to test all steps of the models at once; substitute `/path/to/some/360/image.jpg` with an actual path to an equirectangular image):

    ./360_aware.py /path/to/some/360/image.jpg test_map_eDN.bin --model eDN --mode combined
    ./360_aware.py /path/to/some/360/image.jpg test_map_GBVS.bin --model GBVS --mode combined
    ./360_aware.py /path/to/some/360/image.jpg test_map_SAM.bin --model SAM --mode combined

The basic interface requires 2 positional arguments (input and output files), a `--model` argument, and a `--mode` argument.

# III. Full prediction run

For the arguments to run the models that are described in the paper, see HOW-TO-RUN.txt

You can execute the test code above for images for which the prediction is needed, changing the parameters according to the desired model. 
An example bash for-loop is presented below:

> for file in /path/to/test/images/*.jpg ; do im=`basename $file .jpg`; echo $file ; ./360_aware.py $file /path/to/output/folder/"$im".bin --mode combined --model SAM; done ;

# IV. Best model run 

To predict the saliency map of an equirectangular image with our best model ("combined" interpretation of the input image,
an average saliency map of all three predictors + slight equator bias), run this:

> ./360_aware.py /path/to/input/image.jpg /path/to/output/folder/image.bin --mode combined --model average --centre-bias-weight 0.2

# References

[1] "Graph-based visual saliency",  J. Harel, C. Koch, P. Perona (Advances in Neural Information Processing Systems, 2007) 

[2] "Large-scale optimization of hierarchical features for saliency prediction in natural images", E. Vig, M. Dorr, D. Cox (CVPR'18)

[3] "Predicting human eye fixations via an LSTM-based saliency attentive model",  M. Cornia, L. Baraldi, G. Serra, R. Cucchiara (arXiv:1611.09571)
