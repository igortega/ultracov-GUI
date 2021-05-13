# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:11:46 2021

@author: Ignacio
"""

" Similar image search "

# # ULTRACOV FINDER
# ## LOAD LIBRARIES
import os
import numpy as np
import tensorflow as tf
#from CV_IO_utils import read_imgs_dir
#from CV_transform_utils import apply_transformer, resize_img, normalize_img
#from CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions, plot_img
#from autoencoder3 import AutoEncoder
import matplotlib.pyplot as plt




def normalize(array):
    """ Normalize array to (0, 1) range """
    normalized_array = (array-np.min(array))/(np.max(array)-np.min(array))
    return normalized_array




def encode_frame(frame, model, RES=128):
    """ Preprocess input image to be compared with encoded database 
        INPUT:
            - frame: shaped (x,y) non-square, range (-48, 0)
        OUTPUT:
            - encoded_frame: flattened encoder output shaped (512,) """

    frame_tensor = np.expand_dims(frame, -1) # tensor format (x, y, 1)
    frame_resized = tf.image.resize(frame_tensor, [RES,RES], antialias=True) # (128, 128, 1)
    frame_encoder_input = np.expand_dims(np.array(frame_resized), 0) # array (1, 128, 128, 1)
    norm_frame = normalize(frame_encoder_input)
    encoded_frame = model.predict(norm_frame)
    encoded_frame_flatten = np.reshape(encoded_frame, -1)
    return encoded_frame_flatten



    
def init_similarity():
    """ Loads encoder, display images database, encoded database and 
        frame filenames dictionary.
        INPUT:
            
        OUTPUT:
            - encoder:
            - database_display: numpy array shaped (number_images, height, width)
            - database_encoded: numpy array shaped (number_images, 512)
            - database_filenames: numpy array shaped (number_images,) """
            
    # ## LOAD ENCODER
    inpDir = 'similarity'
    modelName="convAE"
    encoderFile = os.path.join(inpDir, "{}_encoder_v2.h5".format(modelName))   
    encoder = tf.keras.models.load_model(encoderFile)
    
    # ## LOAD IMAGE DATABASE
    database_display = np.load(os.path.join(inpDir, "database_display.npy"))
    print("Database display shape = ", np.shape(database_display))
    print(database_display.min(), database_display.max())
    
    # ## LOAD ENCODED DATABASE
    database_encoded = np.load(os.path.join(inpDir, "database_encoded.npy"))
    print("Encoded database shape = ", np.shape(database_encoded))
    print(database_encoded.min(), database_encoded.max())
    
    # ## LOAD FILENAMES DICTIONARY
    database_filenames = np.load(os.path.join(inpDir, "database_filenames.npy"))
    print("Filenames database shape = ", np.shape(database_filenames))
    
    return encoder, database_display, database_encoded, database_filenames



   
def compare_plot(input_image, similar_image):

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    fig.suptitle("ULTRACOV FINDER", fontsize=16)
    
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title('Input image')
    ax1.axis(False)

    ax2.imshow(similar_image, cmap='gray')
    ax2.set_title('Similar image')
    ax2.axis(False)
    
    fig.savefig('finder_result.png')
    plt.close()

        
        
        
def find_similar(input_frame, encoder, database_encoded):    
    
    # LOAD AND ENCODE INPUT
    encoded_frame = encode_frame(input_frame, model=encoder)
    
    # # FIND THE BEST MATCH
    ETF = tf.keras.losses.cosine_similarity(database_encoded, encoded_frame, axis=-1)
    # print(ETF)
    b = tf.math.argmin(ETF)
    c = tf.keras.backend.eval(b)
    # distancia = np.abs(ETF[c])
    # print(distancia)
    dist = np.array(ETF) # shape: (nimg,)
    indices_sort = np.argsort(dist) # array of indices by closest image
    
    # closest = np.sort(distancias)[:3]
    # i_closest = [np.where(distancias==c) for c in closest]
    
     
    # compare_plot(c, input_frame, database_display)
    
    # match_fname = database_filenames[c]

    return indices_sort
 
# plt.imshow(input_frame)
# plt.title('input')
# plt.show()

# for i in i_closest:
#     plt.imshow(database_display[i][0])
#     plt.show()
