# -*- coding: utf-8 -*-
"""
Search for the closest images in a database
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import ultracov
from ultracov.video_player import normalize


def init_similarity(encoder_filename="convAE_encoder_v2.h5"):
    """Initialize similar image search. Returns the necessary objects for similarity search

    Parameters
    ----------
    encoder_filename:
        Filename of pre-trained encoder model to encode input images with.

    Returns
    -------
    encoder
        Keras model object to encode input images with.
    database_display : array
        Numpy array shaped (number_images, height, width) containing database images in their original format.
    database_encoded : array
        Numpy array shaped (number_images, 512) in encoded vector space.
    database_filenames : array
        Numpy array shaped (number_images,) containing the filenames of database images.
    """

    # Load encoder
    encoder_directory = 'similarity'
    encoder_filepath = os.path.join(ultracov.here, encoder_directory, encoder_filename)
    encoder = tf.keras.models.load_model(encoder_filepath)

    # Load image database
    database_display = np.load(os.path.join(ultracov.here, encoder_directory, "database_display.npy"))
    # print("Database display shape = ", np.shape(database_display))
    # print(database_display.min(), database_display.max())

    # Load encoded database
    database_encoded = np.load(os.path.join(ultracov.here, encoder_directory, "database_encoded.npy"))
    # print("Encoded database shape = ", np.shape(database_encoded))
    # print(database_encoded.min(), database_encoded.max())

    # Load filenames dictionary
    database_filenames = np.load(os.path.join(ultracov.here, encoder_directory, "database_filenames.npy"))
    # print("Filenames database shape = ", np.shape(database_filenames))

    return encoder, database_display, database_encoded, database_filenames


def encode_frame(frame, encoder_model, RES=128):
    """Preprocess input image to be compared with encoded database.

    Parameters
    ----------
    frame : array
        Image shaped (x,y) to perform similarity search on.
    encoder_model
        Pre-trained pre-loaded keras encoder model.
    RES : int
        Required resolution for encoder input.

    Returns
    -------
    encoded_frame_flatten : array
        Encoded frame shaped (512,).
    """

    frame_tensor = np.expand_dims(frame, -1)  # convert to tensor format (x, y, 1)
    frame_resized = tf.image.resize(frame_tensor, [RES, RES], antialias=True)  # to (128, 128, 1)
    frame_encoder_input = np.expand_dims(np.array(frame_resized), 0)  # to (1, 128, 128, 1)
    normalized_frame = normalize(frame_encoder_input)
    encoded_frame = encoder_model.predict(normalized_frame)
    encoded_frame_flatten = np.reshape(encoded_frame, -1)
    return encoded_frame_flatten


def find_similar(input_frame, encoder, database_encoded):
    """Search for closest images to input frame among database.

    Parameters
    ----------
    input_frame : array
        Image to perform similarity search on.
    encoder
        Pre-loaded model to perform similarity search with.
    database_encoded : array
        Numpy array shaped (number_images, 512) in encoded vector space.

    Returns
    -------
    sorted_indices : array
        Sorted array of indices of images in database by similarity (closest image first).
    """

    # Encode input
    encoded_frame = encode_frame(input_frame, encoder_model=encoder)

    # Search for closest images
    ETF = tf.keras.losses.cosine_similarity(database_encoded, encoded_frame, axis=-1)
    distances = np.array(ETF)  # shape: (number images,)
    sorted_indices = np.argsort(distances)  # index of closest image first

    return sorted_indices


def compare_plot(input_frame, similar_image):
    """Makes plot and saves figure of input image and similar image.

    Parameters
    ----------
    input_frame : array
        Image to perform similarity search on.
    similar_image : array
        Chosen image among similarity search output.

    Returns
    -------
    Makes plot and saves figure to root directory.
    """

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    fig.suptitle("ULTRACOV FINDER", fontsize=16)
    
    ax1.imshow(input_frame, cmap='gray')
    ax1.set_title('Input image')
    ax1.axis(False)

    ax2.imshow(similar_image, cmap='gray')
    ax2.set_title('Similar image')
    ax2.axis(False)
    
    fig.savefig('finder_result.png')


if __name__ == "__main__":
    # execution example
    from file_functions import BinFile, Dataset

    # load video and select a frame
    bfile = BinFile(r"C:\Principal\ultracov\videos\4_R1_1.BIN")
    dset = Dataset(bfile, resample=500)
    dset.ScanConvert()
    input_frame = dset.frames[:, :, 0]

    encoder, database_display, database_encoded, database_filenames = init_similarity()

    encoded_frame = encode_frame(input_frame, encoder)

    sorted_indices = find_similar(input_frame, encoder, database_encoded)
    similar_image = database_display[sorted_indices[0], :, :]  # pick best match

    compare_plot(input_frame, similar_image)