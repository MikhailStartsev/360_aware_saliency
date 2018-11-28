#!/usr/bin/env python

import os
import sys
import shutil
import subprocess
import tempfile

from argparse import ArgumentParser
import numpy as np

import h5py
from scipy import io, ndimage

from scipy import misc
from scipy import ndimage
from matplotlib import pyplot as plt

import pickle
from liblinearutil import load_model

from edn_cvpr2014.eDNSalModel import EDNSaliencyModel

PATH = os.path.dirname(os.path.realpath(__file__))
CUBE2SPHERE_PATH = '{}/cube2sphere/cube2sphere.py'.format(PATH)
SPHERE2CUBE_PATH = '{}/sphere2cube/sphere2cube.py'.format(PATH)

GBVS_PATH = '{}/gbvs/'.format(PATH)
SAM_PATH = '{}/saliency_attentive_model/'.format(PATH)

#%% GBVS
def compute_GBVS_saliency_map(img, verbose=False, **kwargs):
    # save image so that Matlab can access it
    img_file = tempfile.NamedTemporaryFile(suffix='.png')
    img_path = img_file.name
    misc.imsave(img_path, img)
    
    out_file = tempfile.NamedTemporaryFile(suffix='.mat')
    # disable blur and undo centre bias via params
    cmd = '''cd {gbvs_path} ; matlab -nojvm -r "gbvs_install(); p = makeGBVSParams(); p.unCenterBias = 1; p.blurfrac = 0.0; [out, ~] = gbvs('{input_fname}', p); saliency_map = out.master_map_resized; save('{output_fname}', ['saliency_map']); exit()"; cd -'''.format(
            gbvs_path=GBVS_PATH, input_fname=os.path.realpath(img_path), output_fname=out_file.name)
    if verbose:
        print >> sys.stderr, cmd
    os.system(cmd)
    res = io.loadmat(out_file)['saliency_map']
    img_file.close()
    out_file.close()
    return res
    
#%% SAM Res-Net
def compute_SAM_saliency_map(img, verbose=False, **kwargs):
    # set the right config file
    proportions = float(img.shape[1]) / img.shape[0]

    mode = 'default'
    if abs(proportions - 2.0) < 1e-5:
        # 2:1
        mode = 'simple'
    elif abs(proportions - 6.0 / 5.0) < 1e-5:
        # 6:5
        mode = 'cutout'
    elif abs(proportions - 1.0) < 1e-5:
        # 1:1
        mode = 'cubeface'
    if verbose:
        print >> sys.stderr, 'Selected mode: {}'.format(mode)
    config_file = '{}/config'.format(SAM_PATH)

    assert mode in {'simple', 'cutout', 'cubeface', 'default'}
    config_file += '_' + mode
    config_file += '.py'
    
    config_dst = '{}/config.py'.format(SAM_PATH)
    shutil.copy(config_file, config_dst)
    
    in_dir = tempfile.mkdtemp() + '/'
    misc.imsave('{}/input.png'.format(in_dir), img)
    
    cmd = '''cd {sam_path} ; THEANO_FLAGS=device=cuda,floatX=float32 python main.py test {input_path} ; cd -'''.format(
            sam_path=SAM_PATH, input_path=in_dir)
    if verbose:
        print >> sys.stderr, cmd
    os.system(cmd)
    
    expected_out_fname = os.path.join(SAM_PATH, 'predictions', 'input.png' + '.h5')
    
    res = h5py.File(expected_out_fname, 'r')['saliency_map'][:]
    
    # clean up
    os.remove(expected_out_fname)
    # restore default config
    shutil.copy('{}/config_default.py'.format(SAM_PATH), config_dst)
    shutil.rmtree(in_dir)
    return res
    
#%%
def compute_eDN_saliency_map(img,
                             no_normalization=True,
                             eDN_desc_path='edn_cvpr2014/slmBestDescrCombi.pkl',
                             eDN_svm_path='edn_cvpr2014/svm-slm',
                             eDN_whitening_params_path='edn_cvpr2014/whiten-slm',
                             verbose=False,
                             **kwargs):
    # determine the image dimensions
    proportions = float(img.shape[1]) / img.shape[0]
    if abs(proportions - 2.0) < 1e-5:
        # 2:1
        insize = (768, 384)
    elif abs(proportions - 1.0) < 1e-5:
        # 1:1
        insize = (512, 512)
    elif abs(proportions - 6.0 / 5.0) < 1e-5:
        # 6:5
        insize = (600, 500)
    elif abs(proportions - 4.0 / 3.0) < 1e-5:
        # 4:3
        insize = (600, 450)
    else:
        # default, 4:3, smaller
        insize = (512, 384)
    if verbose:
        print >> sys.stderr, 'Choosing internal image resolution of', insize
    # read eDN model(s)  
    descFile = open(eDN_desc_path, 'r') 
    desc = pickle.load(descFile) 
    descFile.close() 

    nFeatures = np.sum([d['desc'][-1][0][1]['initialize']['n_filters'] 
                    for d in desc if d != None])

    # load SVM model and whitening parameters 
    svm = load_model(eDN_svm_path) 
    f = open(eDN_whitening_params_path, 'r') 
    white_params = np.asarray([map(float, line.split(' ')) for line in f]).T 
    f.close() 

    # assemble svm model 
    svm_model = {} 
    svm_model['svm'] = svm 
    svm_model['whitenParams'] = white_params 

    bias_to_center_validation = (svm.get_nr_feature() - nFeatures) == 1
    assert bias_to_center_validation == False

    # compute saliency map 
    model = EDNSaliencyModel(descriptions=desc, 
                             svmModel=svm_model, 
                             biasToCntr=False, 
                             insize=insize)  
    sal_map = model.saliency(img, normalize=False)  

    sal_map = sal_map.astype('f')

    if not no_normalization:
        sal_map = (255.0 / (sal_map.max() - sal_map.min()) * 
                   (sal_map - sal_map.min())).astype(np.uint8)
    return sal_map


#%% 360-interpreting saliency map computation
def compute_saliency_map_360(img_360_path,
                             predictor,
                             no_blur=False,
                             mode='simple',
                             verbose=False,
                             cubemap_mode='5-angles-all-axes',
                             **kwargs):
    """
    :param img_360: loaded 360-degree panoram
    :param predictor: a string defining which saliency predictor to use; must be one of the following:
                        'eDN'  (for Ensamble of Deep Networks), 
                        'GBVS' (for Graph Based Visual Saliency), 
                        'SAM'  (for Saliency Attentive Model),
                        'average' (for the saliency map averaged out between eDN, SAM & GBVS)
    :param bo_blur: if True, will not apply blur
    :param mode: one of the following:
                    - 'simple': compute the saliency map for the "frontal" and
                                "back:" view of the scene; combine by taking max()
                    - 'cubemap': a cube map-based interpretation; the input image will be converted to cube 
                                faces and processed individually according to the @cubemap_mode parameter
                    - 'cutout': the cube map face are assembled into a cutout (extended cutout in this case, 
                                see the paper)
                    - 'combined': a combination of 'simple' and 'cubemap' - saliency maps for top and bottom 
                                  cube map faces are computed separately, converted to equirectangular format,
                                  and combined with the two 'simple'-mode saliency maps
                    - 'combined_cutout': same as 'combined', but saliency prediction for top and bottom
                                         cube faces is taken from the 'cutout'-mode style saliency map,
                                         instead of being produced independently
    :param cubemap_mode: only relevant if @mode=='cubemap'; how to avoid too many edges in the resulting
                         equirectangular saliency map:
                            - '1-angle': don't do anything, just predict individual saliency maps and combine directly
                            - '5-angles': create 5 equirectangular saliency maps (as with the '1-angle' option),
                                          differing in the horizontal rotation angle of the cube map; the edges
                                          in the equirectangular maps will be located in different locations in this case,
                                          and combining those with a max() operation will eliminate some of the border artefacts
                            - '5-angles-all-axes': same as '5-angles', but the 5 cube maps are rotated not just around the 
                                                   vertical axis, but the rotations cover all 3 axes. This seems to perform better.
    :param kwargs: arguemnt that will be passed directly to the selected saliency predictor (see @predictor argument)
    """
    
    assert cubemap_mode in {'5-angles', '5-angles-all-axes', '1-angle'}
    
    possible_predictors = ['eDN', 'SAM', 'GBVS']
    if predictor == 'average':
        smaps = [compute_saliency_map_360(img_360_path=img_360_path,
                                          predictor=p,
                                          no_blur=no_blur,
                                          mode=mode,
                                          verbose=verbose,
                                          cubemap_mode=cubemap_mode,
                                          **kwargs) for p in possible_predictors]
        smaps = [write_sal_map(smap, out_fname=None, dry_run=True) for smap in smaps]
        return np.mean(smaps, axis=0)
    
    assert predictor in possible_predictors
    predictor_map = {
            'eDN': compute_eDN_saliency_map,
            'SAM': compute_SAM_saliency_map,
            'GBVS': compute_GBVS_saliency_map
    }
    saliency_predictor = predictor_map[predictor]
    
    if mode != 'simple':  # will need a temporary folder for cubemap images
        tmp_dir = tempfile.mkdtemp()
    else:
        tmp_dir = None
    
    img_360 = misc.imread(img_360_path)
    if mode == 'simple':
        smap_front = saliency_predictor(img_360,
                                        verbose=verbose, **kwargs)
        
        img_back = np.zeros(img_360.shape, dtype=img_360.dtype)
        half_width = img_back.shape[1] / 2
        img_back[:, :half_width, :] = img_360[:, -half_width:, :]
        img_back[:, half_width:, :] = img_360[:, :-half_width, :]
        smap_back = saliency_predictor(img_back,
                                       verbose=verbose, **kwargs)
        
        smap_back_reversed = np.zeros(smap_back.shape)
        smap_back_reversed[:, :half_width] = smap_back[:, -half_width:]
        smap_back_reversed[:, half_width:] = smap_back[:, :-half_width]
        smap_back = smap_back_reversed
        
        res = np.max([smap_front, smap_back], axis=0)
    elif mode == 'cubemap':
        # generate cubemaps
        resolution = 1024
        rotated_sal_maps = []  # save all the rotated saliency maps (avoid border effects)
        
        angle_candidates = []
        if cubemap_mode == '1-angle':
            angle_candidates.append((0, 0, 0))
        elif cubemap_mode == '5-angles':
            angle_candidates += [(0, 0, 0),
                                 (0, 0, 20),
                                 (0, 0, 40),
                                 (0, 0, 60),
                                 (0, 0, 80)]
        elif cubemap_mode == '5-angles-all-axes':
            angle_candidates += [(0, 0, 0),
                                 (0, 0, 45),
                                 (0, 45, 0),
                                 (45, 0, 0),
                                 (45, 45, 0)]
        
        for rotation_x, rotation_y, rotation_z in angle_candidates:
            cubemap_folder = 'cubemap_{}_{}_{}/'.format(rotation_x, 
                                                        rotation_y,
                                                        rotation_z)
            cubemap_folder = os.path.join(tmp_dir, cubemap_folder)
            if not os.path.exists(cubemap_folder):
                os.mkdir(cubemap_folder)
            cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
                   '-o',  cubemap_folder, '-f', 'png',
                   '-t', '3',
                   '-r', str(resolution),
                   '-R', str(rotation_x), str(rotation_y), str(rotation_z)]
            if verbose:
                print >> sys.stderr, ' '.join(cmd)
            subprocess.call(cmd)
            
            # fix the correct order of the cube faces for future reconstruction
            faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                      i,
                                                      resolution) 
                           for i in [3, 1, 4, 2, 6, 5]]
            sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                                i,
                                                                resolution)
                             for i in [3, 1, 4, 2, 6, 5]]
            
            # predict saliency for individual faces
            faces = [misc.imread(face_name) for face_name in faces_order]
            sal_maps = [saliency_predictor(face,
                                           verbose=verbose, **kwargs) 
                        for face in faces]
            
            sal_maps_values = np.vstack(sal_maps)
            sal_maps_min = sal_maps_values.min()
            sal_maps_max = sal_maps_values.max()
            sal_maps = [(smap - sal_maps_min) / (sal_maps_max - sal_maps_min) for smap in sal_maps]  # --> [0; 1]
            
            for face_name, face_sal in zip(sal_map_names, sal_maps):
                misc.toimage(face_sal, cmin=0.0, cmax=1.0).save(face_name)
                # Image.fromarray(face_sal).save(face_name)
            
            # reconstruct the equirectangular image
            out_fname_template = '{}/out####.png'.format(cubemap_folder)
            out_fname = '{}/out0001.png'.format(cubemap_folder)
            cmd = ['python', CUBE2SPHERE_PATH,
                   '-o', out_fname_template, '-f', 'PNG',
                   '-t', '3',
                   '-r', str(img_360.shape[1]), str(img_360.shape[0]),
                   '-R', str(rotation_x), str(rotation_y), str(180 + rotation_z)]
            cmd += sal_map_names
            if verbose:
                print >> sys.stderr, ' '.join(cmd)
            subprocess.call(cmd)
            
            res = misc.imread(out_fname)[:, :, 0].astype(float)
            # res = imageio.imread(out_fname, format='hdr')[:, :, 0]  # should already be float
            
            # reconstruct original values of the saliency map
            # --> [0; 1]
            res -= res.min()
            res /= res.max()
            # --> original values interval
            res *= (sal_maps_max - sal_maps_min)
            res += sal_maps_min
            rotated_sal_maps.append(res)
            
        res = np.max(rotated_sal_maps, axis=0)
    elif mode == 'cutout':
        # generate cubemaps
        resolution = 1024
        rotated_sal_maps = []  # save all the rotated saliency maps (avoid border effects)
        
        # no rotation, since we use the extended cutout anyway
        for rotation_x, rotation_y, rotation_z in [(0, 0, 0)]:
            cubemap_folder = 'cubemap_{}_{}_{}/'.format(rotation_x, 
                                                        rotation_y,
                                                        rotation_z)
            cubemap_folder = os.path.join(tmp_dir, cubemap_folder)
            if not os.path.exists(cubemap_folder):
                os.mkdir(cubemap_folder)
            cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
                   '-o',  cubemap_folder, '-f', 'png',
                   '-t', '3',
                   '-r', str(resolution),
                   '-R', str(rotation_x), str(rotation_y), str(rotation_z)]
            if verbose:
                print >> sys.stderr, ' '.join(cmd)
            subprocess.call(cmd)
            
            # fix the correct order of the cube faces for future reconstruction
            # 0:<front> 1:<back> 2:<left> 3:<right> 4:<top> 5:<bottom>
            faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                      i,
                                                      resolution) 
                           for i in [3, 1, 4, 2, 6, 5]]
            sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                                i,
                                                                resolution)
                             for i in [3, 1, 4, 2, 6, 5]]
            
            faces = [misc.imread(face_name) for face_name in faces_order]
            
            cutout = assemble_cutout(*faces, big=True)
            sal_map = saliency_predictor(cutout,
                                         verbose=verbose,
                                         **kwargs)
            
            sal_maps_min = sal_map.min()
            sal_maps_max = sal_map.max()
            
            # --> [0; 1]
            sal_map -= sal_maps_min
            sal_map /= (sal_maps_max - sal_maps_min)
        
            sal_maps = extract_faces_from_cutout(sal_map, big=True)
            
            for face_name, face_sal in zip(sal_map_names, sal_maps):
                misc.toimage(face_sal, cmin=0.0, cmax=1.0).save(face_name)
            
            # reconstruct the equirectangular image
            out_fname_template = '{}/out####.png'.format(cubemap_folder)
            out_fname = '{}/out0001.png'.format(cubemap_folder)
            cmd = ['python', CUBE2SPHERE_PATH,
                   '-o', out_fname_template, '-f', 'PNG',
                   '-t', '3',
                   '-r', str(img_360.shape[1]), str(img_360.shape[0]),
                   '-R', str(rotation_x), str(rotation_y), str(180 + rotation_z)]
            cmd += sal_map_names
            if verbose:
                print >> sys.stderr, ' '.join(cmd)
            subprocess.call(cmd)
            
            res = misc.imread(out_fname)[:, :, 0].astype(float)
            # res = imageio.imread(out_fname, format='hdr')[:, :, 0]  # should already be float
            
            # reconstruct original values of the saliency map
            # --> [0; 1]
            res -= res.min()
            res /= res.max()
            # --> original values interval
            res *= (sal_maps_max - sal_maps_min)
            res += sal_maps_min
            
            rotated_sal_maps.append(res)
        
        res = np.max(rotated_sal_maps, axis=0)
    elif mode == 'combined':
        # generate cubemaps
        resolution = 1024
       
        cubemap_folder = 'cubemap/'
        cubemap_folder = os.path.join(tmp_dir, cubemap_folder)
        if not os.path.exists(cubemap_folder):
            os.mkdir(cubemap_folder)
        cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
               '-o',  cubemap_folder, '-f', 'png',
               '-t', '3',
               '-r', str(resolution),
               '-R', '0', '0', '0']
        if verbose:
            print >> sys.stderr, ' '.join(cmd)
        subprocess.call(cmd)
        
        # fix the correct order of the cube faces for future reconstruction
        face_bottom_id = 5
        face_bottom_index = [3, 1, 4, 2, 6, 5].index(face_bottom_id)
        face_top_id = 6
        face_top_index = [3, 1, 4, 2, 6, 5].index(face_top_id)
        faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                                  i,
                                                  resolution) 
                       for i in [3, 1, 4, 2, 6, 5]]
        sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                            i,
                                                            resolution) 
                         for i in [3, 1, 4, 2, 6, 5]]
        
        # predict saliency for individual faces
        bottom_face = misc.imread(faces_order[face_bottom_index])
        bottom_sal_map = saliency_predictor(bottom_face, 
                                            verbose=verbose, **kwargs)
        top_face = misc.imread(faces_order[face_top_index])
        top_sal_map = saliency_predictor(top_face, 
                                         verbose=verbose, **kwargs)
        
        if predictor == 'SAM':
            # Divide by the mean disproportion between the max value of the middle 1/3 of the map 
            # and that of the top and bottom 1/3's.
            # The other 2 models have the mean disproportion around 1 (0.95 for eDN and 1.16 for GBVS),
            # and a lower range of its values as well, so nothing changes for them.
            # "Ground truth" maps have the average disproportion of those values at 0.73, actually.
            scaling_factor = 4.51
            bottom_sal_map /= scaling_factor
            top_sal_map /= scaling_factor
            
        all_sal_map_values = np.vstack((top_sal_map, bottom_sal_map)).ravel()
        
        other_sal_map = np.zeros(bottom_sal_map.shape)
        other_sal_map[:] = all_sal_map_values.min()  # level saliency maps
        
        # also need these values to "remap" the image-loaded 360-map values
        sal_maps_min = all_sal_map_values.min()
        sal_maps_max = all_sal_map_values.max()
        
        if not kwargs.get('just_bottom', False):
            sal_maps = [bottom_sal_map if i == face_bottom_index else (top_sal_map if i == face_top_index else other_sal_map)
                        for i in range(6)]
        else:
            sal_maps = [bottom_sal_map if i == face_bottom_index else other_sal_map 
                        for i in range(6)]
            
        sal_maps = [(smap - sal_maps_min) / (sal_maps_max - sal_maps_min) for smap in sal_maps]  #  --> [0; 1]
        
        for face_name, face_sal in zip(sal_map_names, sal_maps):
            misc.toimage(face_sal, cmin=0.0, cmax=1.0).save(face_name)
        
        # reconstruct the equirectangular image
        out_fname_template = '{}/out####.png'.format(cubemap_folder)
        out_fname = '{}/out0001.png'.format(cubemap_folder)
        cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '0', '180']
        cmd += sal_map_names
        if verbose:
            print >> sys.stderr, ' '.join(cmd)
        subprocess.call(cmd)
        
        just_top_bottom_face_360 = misc.imread(out_fname)[:, :, 0].astype(float)
        just_top_bottom_face_360 -= just_top_bottom_face_360.min()
        just_top_bottom_face_360 /= just_top_bottom_face_360.max()  # --> [0; 1]
        # remap to actual values, to be comparable to the directly computed saliency map
        just_top_bottom_face_360 *= (sal_maps_max - sal_maps_min)
        just_top_bottom_face_360 += sal_maps_min            
            
        simple_map = compute_saliency_map_360(img_360_path=img_360_path,
                                              no_blur=True,
                                              mode='simple',
                                              predictor=predictor,
                                              verbose=verbose,
                                              **kwargs)
        
        res = np.max([just_top_bottom_face_360, simple_map], axis=0)
    elif mode == 'combined_cutout':
        # generate cubemaps
        resolution = 1024
        rotated_sal_maps = []  # save all the rotated saliency maps (avoid border effects)
        
        cutout = generate_cutout(img_360_path, out_path=None, big=True)
        cutout_sal_map = saliency_predictor(cutout,
                                            verbose=verbose,
                                            **kwargs)
        sal_map_faces = extract_faces_from_cutout(cutout_sal_map, big=True)
        
        cubemap_folder = 'cubemap/'
        cubemap_folder = os.path.join(tmp_dir, cubemap_folder)
        if not os.path.exists(cubemap_folder):
            os.mkdir(cubemap_folder)
        sal_map_names = ['{}/face_{}_{}_sal_map.png'.format(cubemap_folder,
                                                            i,
                                                            resolution)
                         for i in [3, 1, 4, 2, 6, 5]]
        
        # <front>, <back>, <left>, <right>, <top>, <bottom>
        face_bottom_index = 5
        face_top_index = 4
        bottom_sal_map = sal_map_faces[-1]
        top_sal_map = sal_map_faces[-2]
        
        sal_maps_min = cutout_sal_map.min()
        sal_maps_max = cutout_sal_map.max()
        
        other_sal_map = np.zeros(bottom_sal_map.shape)
        other_sal_map[:] = sal_maps_min  # level saliency maps
        
        # also need these values to "remap" the image-loaded 360-map values
        if not kwargs.get('just_bottom', False):
            sal_maps = [bottom_sal_map if i == face_bottom_index else (top_sal_map if i == face_top_index else other_sal_map)
                        for i in range(6)]
        else:
            sal_maps = [bottom_sal_map if i == face_bottom_index else other_sal_map 
                        for i in range(6)]
        sal_maps = [(smap - sal_maps_min) / (sal_maps_max - sal_maps_min) for smap in sal_maps]  #  --> [0; 1]
        
        for face_name, face_sal in zip(sal_map_names, sal_maps):
            misc.toimage(face_sal, cmin=0.0, cmax=1.0).save(face_name)
        
        # reconstruct the equirectangular image
        out_fname_template = '{}/out####.png'.format(cubemap_folder)
        out_fname = '{}/out0001.png'.format(cubemap_folder)
        cmd = ['python', CUBE2SPHERE_PATH,
               '-o', out_fname_template, '-f', 'png',
               '-t', '3',
               '-r', str(img_360.shape[1]), str(img_360.shape[0]),
               '-R', '0', '0', '180']
        cmd += sal_map_names
        if verbose:
            print >> sys.stderr, ' '.join(cmd)
        subprocess.call(cmd)
        
        just_top_bottom_face_360 = image2map(out_fname)
        just_top_bottom_face_360 -= just_top_bottom_face_360.min()
        just_top_bottom_face_360 /= just_top_bottom_face_360.max()  # --> [0; 1]
        # remap to actual values, to be comparable to the directly computed saliency map
        just_top_bottom_face_360 *= (sal_maps_max - sal_maps_min)
        just_top_bottom_face_360 += sal_maps_min
        
        # now compute the simple map
        simple_map = compute_saliency_map_360(img_360_path=img_360_path,
                                              no_blur=True,
                                              mode='simple',
                                              predictor=predictor,
                                              verbose=verbose,
                                              **kwargs)
        
        res = np.max([just_top_bottom_face_360, simple_map], axis=0)
    else:
        raise ValueError('The operation mode "{}" is not recognized'.format(mode))
    
    if tmp_dir is not None:
        if not verbose:
            shutil.rmtree(tmp_dir)
        else:
            print >> sys.stderr, 'Preserving the images in', tmp_dir    
    
    if not no_blur:
        defatult_sigma = 16.
        default_height = 1024.
        res = ndimage.gaussian_filter(res, sigma=defatult_sigma * img_360.shape[0] / default_height)
        if verbose:
            print >> sys.stderr, 'Picked sigma of', defatult_sigma * img_360.shape[0] / default_height
    return res


#%% Helper functions 
def assemble_cutout(front, back, left, right, top, bottom, big=True):
    n = front.shape[0]  # one side, should all be equal
    if not big:
        if front.ndim == 3:
            cutout = np.zeros((n*3, n*4, 3))
        else:
            cutout = np.zeros((n*3, n*4))
                
        cutout[:n, n:n*2] = top
        cutout[2*n:3*n, n:n * 2] = bottom
        cutout[n:n*2, :n] = right 
        cutout[n:n*2, n:n*2] = back
        cutout[n:n*2, n*2:n*3] = left
        cutout[n:n*2, n*3:n*4] = front
        
        cutout[:n, :n] = np.rot90(top)
        cutout[n*2:n*3, :n] = np.rot90(bottom, k=3)
    
        cutout[:n, n*2:n*3] = np.rot90(top, k=3)
        cutout[n*2:n*3, n*2:n*3] = np.rot90(bottom)
        
        cutout[:n, n*3:n*4] = np.rot90(top, k=2)
        cutout[n*2:n*3, n*3:n*4] = np.rot90(bottom, k=2)
    else:
        if front.ndim == 3:
            cutout = np.zeros((n*5, n*6, 3))
        else:
            cutout = np.zeros((n*5, n*6))
                
        cutout[n:n*2, n*2:n*3] = top
        cutout[n*3:n*4, n*2:n*3] = bottom
        cutout[n*2:n*3, n:n*2] = right 
        cutout[n*2:n*3, n*2:n*3] = back
        cutout[n*2:n*3, n*3:n*4] = left
        cutout[n*2:n*3, n*4:n*5] = front
        
        # rotated top and bottom faces above and below the middle section
        cutout[n:n*2, :n] = np.rot90(top, k=2)
        cutout[n*3:n*4, :n] = np.rot90(bottom, k=2)
        
        cutout[n:n*2, n:n*2] = np.rot90(top)
        cutout[n*3:n*4, n:n*2] = np.rot90(bottom, k=3)    
        cutout[n:n*2, n*3:n*4] = np.rot90(top, k=3)
        cutout[n*3:n*4, n*3:n*4] = np.rot90(bottom)
        cutout[n:n*2, n*4:n*5] = np.rot90(top, k=2)
        cutout[n*3:n*4, n*4:n*5] = np.rot90(bottom, k=2)
        
        cutout[n:n*2, n*5:n*6] = np.rot90(top, k=1)
        cutout[n*3:n*4, n*5:n*6] = np.rot90(bottom, k=3)
        
        # mirrored faces "after" all the "ende" faces
        cutout[n*2:n*3, :n] = front
        cutout[n*2:n*3, n*5:n*6] = right
        
        cutout[:n, n*2:n*3] = front[::-1, ::-1,]
        cutout[n*4:n*5, n*2:n*3] = front[::-1, ::-1]
            
    return cutout

def extract_faces_from_cutout(cutout, big=True, keep_faces=None):
    """
    Returns cube faces in the following order:
        <front>, <back>, <left>, <right>, <top>, <bottom>
    
    :param cutout: cutout image itself; aspect ratio can be either 5x6 or 3x4 
    :big: if True, will assume a 5x6 aspect ratio, otherwise will assume 3x4
    :keep_faces: if not None, must be a list of names (i.e. strings) from 
        the following set: {front, back, left, right, top, bottom}. The values
        for these faces will be left without change, for the others the minimal
        value of the cutout will replace the values in all pixels
    """
    cutout = cutout.copy()
    res = []
    if not big:
        assert cutout.shape[0] / 3 == cutout.shape[1] / 4
        n = cutout.shape[0] / 3
        res.append(cutout[n:n*2, n*3:n*4])  # front
        res.append(cutout[n:n*2, n:n*2])  # back
        res.append(cutout[n:n*2, n*2:n*3])  # left
        res.append(cutout[n:n*2, :n])  # right
        res.append(cutout[:n, n:n*2])  # top
        res.append(cutout[n*2:n*3, n:n*2])  # bottom
    else:
        assert cutout.shape[0] / 5 == cutout.shape[1] / 6
        n = cutout.shape[0] / 5
        res.append(cutout[n*2:n*3, n*4:n*5])  # front
        res.append(cutout[n*2:n*3, n*2:n*3])  # back
        res.append(cutout[n*2:n*3, n*3:n*4])  # left
        res.append(cutout[n*2:n*3, n:n*2])  # right
        res.append(cutout[n:n*2, n*2:n*3])  # top
        res.append(cutout[n*3:n*4, n*2:n*3])  # bottom
        
    if keep_faces is not None:
        face_indices = dict(zip(['front', 'back', 'left', 'right', 'top', 'bottom'], range(6)))
        keep_indices = [face_indices[face_name] for face_name in keep_faces]
        all_faces_min = np.vstack(res).min()
        for i in range(len(res)):
            if i not in keep_indices:
                res[i][:] = all_faces_min
    return res

def image2map(img_path):
    res = misc.imread(img_path)
    if res.ndim == 3:
        res = res[:, :, 0]
    return res.astype(float)

def generate_cutout(img_360_path, out_path, big=True, 
                    rotation_x=0, rotation_y=0, rotation_z=0,
                    verbose=False):
    cubemap_folder = tempfile.mkdtemp() + '/'
    resolution = 1024
    cmd = ['python', SPHERE2CUBE_PATH, img_360_path, 
           '-o',  cubemap_folder, '-f', 'png',
           '-t', '3',
           '-r', str(resolution),
           '-R', str(rotation_x), str(rotation_y), str(rotation_z)]
    if verbose:
        print >> sys.stderr, ' '.join(cmd)
    subprocess.call(cmd)
    
    # fix the correct order of the cube faces for future reconstruction
    # 0:<front> 1:<back> 2:<left> 3:<right> 4:<top> 5:<bottom>
    faces_order = ['{}/face_{}_{}.png'.format(cubemap_folder,
                                              i,
                                              resolution) 
                   for i in [3, 1, 4, 2, 6, 5]]    
    faces = [misc.imread(face_name) for face_name in faces_order]
    
    shutil.rmtree(cubemap_folder)
    
    cutout = assemble_cutout(*faces, big=True)
    if out_path is not None:
        # the only place where it is acceptable to use misc.imsave: this is used on uint8 images, should be fine
        misc.imsave(out_path, cutout)
    return cutout

def write_sal_map(sal_map, out_fname, save_png_too=True, dry_run=False):
    sal_map = sal_map.astype(np.float32)
    sal_map -= sal_map.min()
    if save_png_too and not dry_run:
        # save the visualization too
        # just a visualizaion, so we don't really worry about normalization
        plt.imsave(out_fname + '.png', sal_map / sal_map.max(), cmap='jet')
    sal_map /= sal_map.sum()
    if not dry_run:
        sal_map.tofile(out_fname)
    return sal_map
    
#%% Interface
def parse_args():
    parser = ArgumentParser('360-aware saliency model toolbox')
    
    parser.add_argument('input', 
                        help='Path to 360-equirectangular RGB input image')
    parser.add_argument('binary_output',
                        help='Path to the .bin output file. '
                             'Will write the resulting saliency map to this '
                             'file as double saliency values aranged row-wise. '
                             'The sum will be equal to 1, minimal value is 0. '
                             'Will additionally create a similar .png file with the saliency map visualisation.')
    parser.add_argument('--mode', default='direct', 
                        choices=['simple', 'cubemap', 'combined', 'cutout', 
                                 'combined_cutout'],
                        help='Which mode the 360 image interpretation to use')
    parser.add_argument('--model', default='eDN', 
                    choices=['eDN', 'SAM', 'GBVS', 'average'],
                    help='Which model to use for prediction. More details in README.')
    parser.add_argument('--cubemap-mode', default='5-angles-all-axes', 
                        choices=['1-angle', '5-angles', '5-angles-all-axes'],
                        help='debug parameter; do not change')
    parser.add_argument('--verbose', '-v', default=False,
                        action='store_true',
                        help='''Whether to output runtime info. Some details will be outputed either way, 
                        but you can squash them by adding "2>/dev/null 1>/dev/null", without quotes, to the end of your ./360_aware.py ... command.''')
    parser.add_argument('--just-bottom', 
                        action='store_true',
                        help='debug parameter; do not set')
    parser.add_argument('--centre-bias-weight',
                        '--center-bias-weight',
                        type=float,
                        default=0.0,
                        help='The weight of the added centre bias (disabled by default)')
    return parser.parse_args()

def main(): 
    args = parse_args()
    
    sal_map = compute_saliency_map_360(img_360_path=args.input,
                                       mode=args.mode,
                                       predictor=args.model,
                                       verbose=args.verbose,
                                       just_bottom=args.just_bottom,
                                       cubemap_mode=args.cubemap_mode)
    
    if args.centre_bias_weight != 0:
        mean_map_shape = (512, 1024)
        mean_map = np.fromfile('HeadEye_mean_sal_map_all.bin', dtype=np.float32).reshape(mean_map_shape)
        zoom = float(sal_map.shape[0]) / mean_map_shape[0]
        mean_map = ndimage.interpolation.zoom(mean_map, zoom)
        mean_map /= mean_map.sum()
        sal_map += mean_map * args.centre_bias_weight
    
    write_sal_map(sal_map, args.binary_output)

if __name__ == "__main__": 
    main() 
