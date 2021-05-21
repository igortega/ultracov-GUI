# -*- coding: utf-8 -*-
"""
Functions for video display
"""

import PySimpleGUI as sg
from PIL import Image
import io
import os
import numpy as np

from ultracov.file_functions import BinFile, Dataset


def normalize(array, norm='unit'):
    """Normalize array to (0,1) depth interval

    Parameters
    ----------
    array
        Array of arbitrary depth to be normalized to (0,1) or (0, 255)
    norm : str
        Kind of normalization (unit as default)

    Returns
    -------
    normalized_array
        Same array normalized to (0,1) or (0,255)

    """

    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    if norm == '255':
        normalized_array *= 255
    return normalized_array


def get_img_data(frame):
    """Convert image to bytes data

    Parameters
    ----------
    frame
        Numpy array of image

    Returns
    -------
    bytes data of image

    """

    img = Image.fromarray(frame, mode='L')
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def load_video_data(filepath, aspect='sector'):
    """Loads data from .bin video in filepath and returns bytes data, BinFile and Dataset objects.

    Parameters
    ----------
    filepath : str
        Path to .bin video to be loaded.
    aspect : str
        Aspect to load video in. 'sector' (default) or 'square'.

    Returns
    -------
    bytes_images
    bfile
    dset
    """

    bfile = BinFile(filepath)
    dset = Dataset(bfile, resample=380)

    if aspect == 'sector':
        dset.ScanConvert()
        frames = np.int8(normalize(dset.frames, norm='255'))
        bytes_images = [get_img_data(frames[:, :, i]) for i in range(dset.nimg)]
    if aspect == 'square':
        frames = np.int8(normalize(dset.bscan, norm='255'))
        bytes_images = [get_img_data(frames[:, :, i]) for i in range(dset.nimg)]

    return bytes_images, bfile, dset


def play_video(video_path, aspect='sector'):
    """Play single video on 'sector' or 'square' format

    Parameters
    ----------
    video_path : str
        Path to .bin video to be loaded
    aspect
        Aspect to load video in. 'sector' (default) or 'square'.

    Returns
    -------
    Displays video.
    """

    video_data, bfile_data, dset_data = load_video_data(video_path, aspect=aspect)

    image_display = sg.Image(data=video_data[0], key='display')

    layout = [[image_display]]

    window = sg.Window('Video player', layout)

    i = 0
    while True:
        n_frame = i % len(video_data)
        event, values = window.read(timeout=100)

        if event == sg.WIN_CLOSED:
            break

        # Update displayed image
        window['display'].update(data=video_data[n_frame])
        i += 1


def video_player(videos_dir, aspect='sector'):
    """

    Parameters
    ----------
    videos_dir
    aspect

    Returns
    -------

    """
    video_list = [fname for fname in os.listdir(videos_dir) if os.path.splitext(fname)[-1] in ('.bin', '.BIN')]
    selected_video = video_list[0]

    video_data = {}
    bfile_data = {}
    dset_data = {}

    video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(
        os.path.join(videos_dir, selected_video), aspect=aspect)

    videos_listbox = [sg.Listbox(values=video_list,
                                 default_values=selected_video,
                                 key='listbox',
                                 size=(30, 20),
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
        n_frame = i % len(video_data[selected_video])
        event, values = window.read(timeout=50)
        print(event, values)
        if event == sg.WIN_CLOSED:
            break

        # Select new video
        if event == 'listbox':
            # Select new video
            selected_video = values['listbox'][0]
            if not selected_video in video_data.keys():
                video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(
                    os.path.join(videos_dir, selected_video), aspect=aspect)

        # Update displayed image
        window['display'].update(data=video_data[selected_video][n_frame])
        i += 1
