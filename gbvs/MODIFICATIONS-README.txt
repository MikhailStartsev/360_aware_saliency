gbvs.m was modified to not call `mat2gray` on the variables "master_map" and "master_map_resized"
(also to not rescale the values in these maps).

This is important if we want to compare and combine the values in 2 saliency maps produced by 
the GBVS toolbox.
