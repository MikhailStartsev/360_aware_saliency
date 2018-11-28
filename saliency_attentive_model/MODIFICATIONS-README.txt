Added the following flags to utilities.py/postprocess_predictions()
    :no_normalization (False by default) 
    :no_blur (False by default)

to remove any postprocessing on the network's side (important when we work with more than 1 image of the same scene).

Added appropriate flags in main.py.
Saving not to image, but to an .h5 file (to avoid uint8-conversion along the way).
