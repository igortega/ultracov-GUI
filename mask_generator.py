# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:48:38 2021

@author: Ignacio
"""

""" Loads trained model from .h5 file. 
    Takes images as input.
    Returns predicted pleura masks as output
"""

import time
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io





def load_model(model_path):
    """ Load trained segmentator model
        INPUT:
            - model_path: path to .h5 file
        OUTPUT:
            - model    """
    start_time = time.time()
    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Size: %.2f MB" % (os.path.getsize(model_path)/1024/1024))
    print("Model loading time: %.2f s" % (time.time()-start_time))
    
    return model



 
def load_img(dset_frame):
    """ Returns image in suitable format to feed model
        INPUT:
            - dset_frame: bidimensional numpy array to make prediction of. Range (-48, 0)
        OUTPUT: 
            - img_expanded: image in suitable format for model input  """
    
    # Resize (RES,RES)
    RES = 64
    img_resized = cv2.resize(dset_frame, (RES,RES)) 
    # Normalize to (0,1) interval
    img_norm = (img_resized-np.min(img_resized))/(np.max(img_resized)-np.min(img_resized))
    # Expand dimensions to (1, RES, RES, 1)
    img_expanded = np.expand_dims(np.expand_dims(img_norm, axis=0), axis=3)
    # print(img_expanded.shape)
    # input_img = img_expanded/65536 # 16 bit normalization
    # input_img = img_expanded/48 + 1 # dB normalization
    
    return img_expanded




def predict(input_img, model):
    """ Feeds input_img to model and returns mask prediction
        INPUT:
            - input_img: numpy array in suitable format and normalization
            - model: segmentator model to make prediction with
        OUTPUT:
            - output_img: mask prediction for input_img in (64, 64) shape   """
    # start_time = time.time()
    prediction = model.predict(input_img)
    output_img = prediction[0][:,:,0]
    # print("Mask generation time: %.2f s" % (time.time()-start_time))
    
    return output_img




def blend_mask(img, mask):
    """ Blend original image and predicted pleura mask in a single image
        INPUT:
            -img: array of ground image
            -mask: array of predicted pleura mask
        OUTPUT:
            -bimg: blended image in bytes for display """
            
    mask = 255 - mask # Reverse mask
    
    hlight = Image.new(mode='RGB',
                       size=(img.shape[1], img.shape[0]), 
                       color=(255,255,0))   # Yellow highlight
    
    a = Image.fromarray(img).convert('RGB')
    b = Image.fromarray(mask)
    
    mix = Image.composite(a, hlight, b) # Generate blended image
    # Convert to bytes
    bio = io.BytesIO()
    mix.save(bio, format="PNG")
    bimg = bio.getvalue()
    return bimg




# def analyze_mask(mask):
#     mask = np.int32(np.round(np.squeeze(mask)))
#     mask_labels = measure.label(mask)
#     mask_regions = measure.regionprops(mask_labels)
#     mask_regions.sort(key=lambda x: x.area, reverse=True)
    
#     # if len(regions) > 0:
#     #     mask_info = [len(regions), regions[0].area, regions[0].centroid[0], regions[0].centroid[1]]
#     # else:
#     #     mask_info = [np.nan, np.nan, np.nan, np.nan]
#     return mask_labels, mask_regions




def plot_result(img, mask):
    """ Show input image and predicted mask """
    
    if not img.shape == mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.cla()
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax1.set_title('input image')
    
    ax2.cla()
    ax2.imshow(mask, cmap='gray')
    ax2.axis('off')
    ax2.set_title('predicted mask')




# if __name__=="__main__":
#     model_path = 'pleura_model.h5'
#     video_path = r"C:\Principal\ultracov\videos\4_R1_1.BIN"   
    
#     # Get input image from video
#     bfile = BinFile(video_path)
#     dset = Dataset(bfile, resample=600)
#     dset.ScanConvert() # Change to sector format
    
