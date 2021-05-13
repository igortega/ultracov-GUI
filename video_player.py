# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:58:48 2021

@author: Ignacio
"""

" Video display functions "

import PySimpleGUI as sg
from PIL import Image
import cv2
import io
import os
import numpy as np

from file_functions import BinFile, Dataset



def get_img_data(frame):
    """ Generate image bytes data
        INPUT:
            - frame: numpy array of image
        OUTPUT:
            - bytes of image """
            
    img = Image.fromarray(frame, mode='L')
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()




def bscan_resize(bfile, size=(200,200)):
    """ Reshapes and resizes video data contained in BinFile object
        INPUT:
            - bfile: BinFile object of video to be loaded 
            - size: tuple of desired frame dimensions 
        OUTPUT:
            - resized_frames: numpy array of dimensions (frame, xpixel, zpixel) """
    
    scan_data = bfile.scan_data[0]
    frames = np.reshape(scan_data, (bfile.trigger_lines_number, bfile.n_ascan[0], bfile.n_samples[0]))
    resized_frames = np.array([cv2.resize(frames[i,:,:], dsize=size) for i in range(bfile.trigger_lines_number)])
    return resized_frames




def load_video_data(filepath, aspect='sector'):
    """ Load .bin file in filepath. 
        Returns BinFile and Dataset objects.
        Returns video data in bytes to be displayed in GUI.
        INPUT:
            - filepath: path to .bin file to be loaded
            - aspect: video format (square/sector)
        OUTPUT:
            - bfile: BinFile object 
            - dset: Dataset object (if aspect='sector')
            - bimgs: list of video frames in bytes """
            
    bfile = BinFile(filepath)
    if aspect == 'sector':
        dset = Dataset(bfile, resample=380)
        dset.ScanConvert()
        frames = np.int8(dset.frames/48*255 + 255)
        bimgs = [get_img_data(frames[:,:,i]) for i in range(dset.nimg)]
    if aspect == 'square':
        frames = bscan_resize(bfile)
        frames = np.int8(frames/480*255 + 255)
        bimgs = [get_img_data(frames[i,:,:].T) for i in range(bfile.trigger_lines_number)]
    
    try:
        return bimgs, bfile, dset
    except:
        raise Exception('Square format does not return Dataset object')
        
        


def video_player(videos_dir, aspect='sector'):        
    video_list = [fname for fname in os.listdir(videos_dir) if os.path.splitext(fname)[-1] in ('.bin', '.BIN')]
    selected_video = video_list[0]
    
    
    video_data = {}
    bfile_data = {}
    dset_data = {}
    # video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(os.path.join(videos_dir, selected_video))
    video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(os.path.join(videos_dir, selected_video), aspect=aspect)
    
    
    videos_listbox = [sg.Listbox(values=video_list,
                                 default_values=selected_video,
                                key='listbox', 
                                size=(30,20), 
                                enable_events=True, 
                                font=('helvetica', 15), 
                                auto_size_text=True)]
    
    image_display = [sg.Image(data=video_data[selected_video][0], 
                                 key='display')]
    
    video_selection_column = sg.Column([videos_listbox])
    
    video_display_column = sg.Column([image_display])
    
    layout = [[video_selection_column, video_display_column]]
    
    window = sg.Window('Herramienta etiquetado ULTRACOV', 
                       layout)
    
    
    i = 0
    while True:
        n_frame = i%len(video_data[selected_video])
        event, values = window.read(timeout=50)
        print(event, values)
        if event == sg.WIN_CLOSED:
            break
        
        # Select new video
        if event == 'listbox':
            # Select new video
            selected_video = values['listbox'][0]
            if not selected_video in video_data.keys():
                # video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(os.path.join(videos_dir, selected_video))
                video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(os.path.join(videos_dir, selected_video), aspect=aspect)
    
                
        #### Update displayed image
        window['display'].update(data=video_data[selected_video][n_frame])
        i += 1




def play_video(video_path, aspect='sector'):
    """ Play single video on 'sector' or 'square' format """
    
    video_data, bfile_data, dset_data = load_video_data(video_path, aspect=aspect)
    
    image_display = sg.Image(data=video_data[0], key='display')
    
    layout = [[image_display]]
    
    window = sg.Window('Herramienta etiquetado ULTRACOV', layout)
    
    
    i = 0
    while True:
        n_frame = i%len(video_data)
        event, values = window.read(timeout=100)
        
        if event == sg.WIN_CLOSED:
            break

        #### Update displayed image
        window['display'].update(data=video_data[n_frame])
        i += 1