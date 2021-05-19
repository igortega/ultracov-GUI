# -*- coding: utf-8 -*-
"""Generate pleura masks through neural network segmentation

Pre-trained models are located in 'pleura' directory as .h5 files.
Predictions can be made on both 'square' and 'sector' format images.
A frame of a video is fed as input and its predicted pleura mask
is returned as output.
"""

import time
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io

from file_functions import BinFile, Dataset


def load_model(model_path):
    """Load pre-trained segmentation model

    Parameters
    ----------
    model_path
        Path to selected model

    Returns
    -------
    model
        Keras model object to be fed to predict() function
    """

    start_time = time.time()  # get model loading time
    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Size: %.2f MB" % (os.path.getsize(model_path)/1024/1024))  # get model size
    print("Model loading time: %.2f s" % (time.time()-start_time))

    return model


def prepare_for_predict(input_image):
    """Convert image to proper format to be fed to segmentation model

    Parameters
    ----------
    input_image : array
        Image to make mask prediction of in either 'square' or 'sector' format.

    Returns
    -------
    prepared_image : array
        Processed image ready to be fed to predict()
    """
    
    # Resize (RES,RES)
    RES = 64
    resized_image = cv2.resize(input_image, (RES, RES))
    # Normalize to (0,1) interval
    minimum = np.min(resized_image)
    maximum = np.max(resized_image)
    normalized_image = (resized_image - minimum) / (maximum - minimum)
    # Expand dimensions to (1, RES, RES, 1)
    prepared_image = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=3)

    return prepared_image


def predict(prepared_image, model):
    """Make pleura mask prediction of an image through a given model.

    Parameters
    ----------
    prepared_image : array
        Properly formatted and normalized image to make prediction of.
    model
        Preloaded chosen model through load_model() function.

    Returns
    -------
    mask : array
        Image of predicted pleura mask.
    """

    # start_time = time.time()
    mask = model.predict(prepared_image, )[0][:, :, 0]
    # print("Mask generation time: %.2f s" % (time.time()-start_time))
    
    return mask


def blend_mask(input_image, mask):
    """Generate bytes image with blended highlighted mask

    Parameters
    ----------
    input_image : array
        Image to make mask prediction of in either 'square' or 'sector' format.
    mask : array
        Image of predicted pleura mask.

    Returns
    -------
    bytes_blended_image
        Blend of input image and mask in bytes format (suitable for GUI display).
    """

    blend_size = (input_image.shape[1], input_image.shape[0])

    highlight = Image.new(mode='RGB',
                          size=blend_size,
                          color=(255, 255, 0))   # Yellow highlight

    mask = cv2.resize(src=255*mask, dsize=blend_size)  # mask originally normalized to (0,1)
    mask = 255 - mask  # Reverse mask

    # Normalize input image to (0,255)
    minimum = np.min(input_image)
    maximum = np.max(input_image)
    normalized_image = 255*((input_image - minimum) / (maximum - minimum))

    # Generate Image objects
    image = Image.fromarray(normalized_image).convert('RGB')
    mask = Image.fromarray(mask).convert('L')

    blended_image = Image.composite(image, highlight, mask)  # Generate blended image

    # Convert to bytes
    bio = io.BytesIO()
    blended_image.save(bio, format="PNG")
    bytes_blended_image = bio.getvalue()

    return bytes_blended_image


if __name__ == "__main__":
    # Example
    sector_model_path = r'pleura\pleura_sector_model.h5'
    video_path = r"C:\Principal\ultracov\videos\4_R1_1.BIN"

    # Load pretrained model
    model = load_model(sector_model_path)

    # Get input image from video
    bfile = BinFile(video_path)
    dset = Dataset(bfile, resample=600)
    dset.ScanConvert()  # Convert to sector format
    input_sector_image = dset.frames[:, :, 0]

    # Preprocess input image
    prepared_image = prepare_for_predict(input_sector_image)

    # Get mask
    mask = predict(prepared_image, model)

    # Plot result
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(input_sector_image)
    axs[1].imshow(mask)