# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:12:59 2021

@author: Ignacio
"""

""" Interfaz para etiquetar exploraciones y generar base de datos """

import PySimpleGUI as sg
import cv2
import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from urllib.request import urlretrieve

from motion_detector_bin import motion_detection
from mask_generator import load_model, load_img, predict, blend_mask
from similarity import init_similarity, find_similar
from frame_quality import analyze_frame, is_valid
from video_player import get_img_data, load_video_data


##  AUXILIAR FUNCTIONS
def get_files():
    # download required files
    url = r"http://x-cov.com/data/ULTRACOV_UCM_V0.zip"
    print('Downloading required files...')
    urlretrieve(url, 'required_files.zip')

    zfile = ZipFile('required_files.zip')
    # extract files to each directory
    directories = ['pleura', 'similarity']
    for d in directories:
        requirements_path = os.path.join(d, 'file_requirements.txt')
        with open(requirements_path) as f:
            files = f.readlines()
        for f in files:
            fname = f.split('\n')[0]
            zfile.extract(fname, path=d)

    zfile.close()


def get_score(database):
    """ Calculate score of a labelled region and updates database
        INPUT:
            - database: DataFrame with video labels """

    score3 = bool( sum( [int(database[key]) for key in ['effusion', 'consolidation-', 'consolidation+']] ) )
    score2 = bool( sum( [int(database[key]) for key in ['confluent+']] ) )
    score1 = bool( sum( [int(database[key]) for key in ['blines', 'confluent-']] ) )
    score0 = bool( sum( [int(database[key]) for key in ['alines']] ) )
    empty = sum(list(map(int, database.values()))) == 0

    score_list = [score3, score2, score1, score0, empty]
    score = 3 - score_list.index(True)
    # database.loc[empty, 'score'] = 0
    # database.loc[score0, 'score'] = 0
    # database.loc[score1, 'score'] = 1
    # database.loc[score2, 'score'] = 2
    # database.loc[score3, 'score'] = 3
    return score


def load_database(selected_video):
    """ Load previously saved labels and comment or generate new ones if they do not exist """
    # label_list = [fname for fname in os.listdir(labels_dir) if os.path.splitext(fname)[-1] == '.csv']

    database_filename = os.path.splitext(selected_video)[0] + '.csv'
    comment_filename = os.path.splitext(selected_video)[0] + '.txt'

    database_filepath = os.path.join(labels_dir, database_filename)
    comment_filepath = os.path.join(labels_dir, comment_filename)

    keys = label_dict['key']
    # Previously saved label database
    if os.path.exists(database_filepath):
        with open(database_filepath) as f:
            labels_data = f.readlines()
        values = labels_data[1].split('\n')[0].split(';')
        values = [int(v) for v in values]

        with open(comment_filepath) as f:
            try:
                comment = f.readlines()[0]
            except:
                comment = ''

        previous = True  # previous data exists

    # Blank database
    else:
        values = [0 for k in range(len(keys))]
        # database.to_csv(database_filepath, sep=';')
        comment = ''
        previous = False  # previous data does not exist


    database = dict(zip(keys, values))

    if previous == False:  # generate blank labels and comment files
        save_database(selected_video, database, comment)

    return database, comment, previous


def save_database(selected_video, database, comment):
    database_filepath = 'labels/' + os.path.splitext(selected_video)[0] + '.csv'  # save labels
    comment_filepath = 'labels/' + os.path.splitext(selected_video)[0] + '.txt'  # save comment

    keys = label_dict['key']
    with open(database_filepath, 'w') as f:
        keys_line = ';'.join(keys)
        values = list(database.values())
        values_line = ';'.join(list(map(str,values)))
        f.writelines([keys_line + '\n', values_line])
        f.close()

    with open(comment_filepath, 'w') as f:
        f.write(comment)



if __name__ == '__main__':

    ###  INITIALIZE
    # Get required files
    if not os.path.exists('required_files.zip'):
        get_files()

    # Read list of .bin videos in selected directory
    videos_dir = 'videos'
    try:
        video_list = os.listdir(videos_dir)
    except:
        videos_dir = sg.popup_get_folder('Open video directory')

    video_list = [fname for fname in os.listdir(videos_dir) if os.path.splitext(fname)[-1] in ('.bin', '.BIN')]

    # Create labels directory / initialize previous labels
    labels_dir = 'labels'
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)

    # Define labels
    label_dict = {'name': ['A-lines',
                           'Isolated B-lines',
                           'Confluent B-lines <50%',
                           'Confluent B-lines >50%',
                           'Pleural effusion',
                           'Subpleural consolidation',
                           'Consolidation'],
                  'key': ['alines',
                          'blines',
                          'confluent-',
                          'confluent+',
                          'effusion',
                          'consolidation-',
                          'consolidation+']}

    label_color = ['#00fe00',
                   '#fffe00',
                   '#fffe00',
                   '#ff7900',
                   '#fe0000',
                   '#fe0000',
                   '#fe0000']

    # Initialize label dataframe
    # cols = label_dict['key'].copy()
    # cols.append('score')
    # cols.append('comment')
    # label_df = pd.DataFrame(columns=cols, index=[0])
    # label_df.iloc[:, :] = 0
    # label_df.loc[0, 'comment'] = ''

    # Create variables to hold video data
    video_data = {}
    bfile_data = {}
    dset_data = {}

    # Initialiaze pleura segmentation and similarity models
    pleura_model = load_model('pleura/pleura_model.h5')
    pleura_square_model = load_model('pleura/pleura_square_model.h5')
    encoder, database_display, database_encoded, database_filenames = init_similarity()

    ##  WINDOW LAYOUT
    videos_listbox = [sg.Listbox(values=video_list,
                                 key='listbox',
                                 size=(30, 20),
                                 enable_events=True,
                                 font=('helvetica', 15),
                                 auto_size_text=True)]

    score_display = [sg.Text('Region score:',
                     auto_size_text=True,
                     font=('helvetica', 15)),
             sg.Text(str('-'),
                     key='region-score',
                     auto_size_text=True,
                     font=('helvetica', 15))]

    image_display = [sg.Image(key='display',
                              filename='logo.png',
                              background_color='#eeeeee')]

    quality_bar = [sg.T('Image quality'), sg.ProgressBar(1,
                                                         size=(15, 10),
                                                         key='quality',
                                                         bar_color=['red', 'green'])]

    frame_buttons = [sg.Button('Previous',
                               key='prev',
                               font=('helvetica', 15),
                               auto_size_button=True),
                     sg.Button('Next',
                               key='next',
                               font=('helvetica', 15),
                               auto_size_button=True)]

    analysis_buttons = [sg.Button('Motion detection',
                                  key='motion',
                                  disabled=True),
                        sg.Button('Pleura detection',
                                  key='pleura_detect',
                                  disabled=True),
                        sg.Button('Similar images',
                                  key='similar',
                                  disabled=True),
                        sg.Button('Test mode',
                                  key='test',
                                  disabled=True)]

    label_checkboxes = [[sg.Checkbox(text=label_dict['name'][i],
                                     key=label_dict['key'][i],
                                     enable_events=True,
                                     disabled=True,
                                     checkbox_color=label_color[i],
                                     text_color='#000000',
                                     background_color='#dddddd',
                                     font=('helvetica', 20),
                                     auto_size_text=True)] for i in range(len(label_dict['name']))]

    comment_box = [sg.Button('Save',
                             key='save',
                             font=('helvetica', 15),
                             auto_size_button=True),
                   sg.Input(key='comment',
                            size=(60, 60),
                            font=('helvetica', 10))]

    video_selection_column = sg.Column([videos_listbox,
                                        frame_buttons])

    video_display_column = sg.Column([quality_bar,
                                      image_display,
                                      analysis_buttons],
                                     element_justification='center',
                                     background_color='#eeeeee')

    label_column = [score_display] + label_checkboxes + [comment_box]
    # label_checkboxes.insert(0, score)
    # label_checkboxes.append(comment_box)

    label_column = sg.Column(label_column)

    layout = [[video_selection_column, video_display_column, label_column]]

    window = sg.Window('Herramienta etiquetado ULTRACOV',
                       layout)


    def test_mode(true_labels):
        label_checkboxes = [[sg.Checkbox(text=label_dict['name'][i],
                                         key=label_dict['key'][i],
                                         enable_events=True,
                                         checkbox_color=label_color[i],
                                         text_color='#000000',
                                         background_color='#dddddd',
                                         font=('helvetica', 20),
                                         auto_size_text=True)] for i in range(len(label_dict['name']))]
        check_button = [sg.Button('Check',
                                  key='check')]
        label_checkboxes.append(check_button)

        layout = label_checkboxes
        test_window = sg.Window("Test mode", layout, modal=True)
        # choice = None
        while True:
            event, values = test_window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == 'check':
                for label in values.keys():
                    if not values[label] == true_labels[label]:
                        test_window[label].update(background_color='#ff0000')
                    else:
                        test_window[label].update(background_color='#00ff00')
        test_window.close()


    def similar_mode(similar_indices, database_display, database_filenames, n_images=4):
        """ Launch window for similar images display """

        indices = similar_indices[:n_images]
        similar_images_fnames = list(database_filenames[indices])
        similar_images = database_display[indices, :, :]
        similar_images = np.int8(
            255 * ((similar_images - similar_images.min()) / (similar_images.max() - similar_images.min())))
        similar_images_display = []
        for k in range(n_images):
            img = get_img_data(similar_images[k, :, :])
            similar_images_display.append(img)

        selected_image = 0
        ### DEFINE LAYOUT
        image_list = [[sg.Listbox(values=similar_images_fnames,
                                  default_values=similar_images_fnames[selected_image],
                                  size=(15, 20),
                                  enable_events=True,
                                  key='similar-list')]]

        image_display = [[sg.Image(data=similar_images_display[selected_image],
                                   key='similar-display')]]

        similar_labels = [[sg.Checkbox(text=label_dict['name'][i],
                                       key=label_dict['key'][i],
                                       disabled=True,
                                       checkbox_color=label_color[i],
                                       text_color='#000000',
                                       background_color='#dddddd',
                                       font=('helvetica', 20),
                                       auto_size_text=True)] for i in range(len(label_dict['name']))]

        layout = [[sg.Column(image_list), sg.Column(image_display), sg.Column(similar_labels)]]
        similar_window = sg.Window('Similarity', layout, modal=True)

        while True:
            event, values = similar_window.read(timeout=100)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == 'similar-list':
                print(values)
                print(values[event][0])
                selected_image = similar_images_fnames.index(values[event][0])
                update_img = similar_images_display[selected_image]
                similar_window['similar-display'].update(data=update_img)

        similar_window.close()


    ##  MAIN WINDOW
    pleura_mode = 'off'
    i = 0
    selected_video = None
    database = None
    n_frame = 0
    previous_data = None

    while True:
        event, values = window.read(timeout=100)
        # print(event, values)
        if event == sg.WIN_CLOSED:
            break

        # Activate label checkboxes and buttons
        if not selected_video == None:
            for key in label_dict['key']:
                window[key].update(disabled=False)
            window['motion'].update(disabled=False)
            window['pleura_detect'].update(disabled=False)
            window['similar'].update(disabled=False)
            window['test'].update(disabled=False)

        # Tick checkbox
        if event in label_dict['key']:
            # database.loc[selected_video, event] = int(values[event])
            database[event] = int(values[event])  # update database
            # Update score
            window['region-score'].update(str(get_score(database)))

        # Select new video
        if event == 'listbox':
            selected_video = values['listbox'][0]
            n_frame = 0

            # Load video data
            if not selected_video in video_data.keys():
                video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(
                    os.path.join(videos_dir, selected_video))

            # Load video labels
            database, comment, previous_data = load_database(selected_video)

            # Load labels on checkboxes, score and comment
            window.fill(database)
            window['region-score'].update(str(get_score(database)))
            window['comment'].update(value=comment)

        # # Go to previous/next video
        # if event in ('prev', 'next'):
        #     if event == 'prev':
        #         n_change = -1
        #     if event == 'next':
        #         n_change = +1
        #
        #     # Select new video
        #     n_selection = video_list.index(selected_video)
        #     n_selection = (n_selection + n_change) % len(video_list)
        #     selected_video = video_list[n_selection]
        #     window['listbox'].set_value(selected_video)
        #     n_frame = 0
        #
        #     # Load video data
        #     if not selected_video in video_data.keys():
        #         video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(
        #             os.path.join(videos_dir, selected_video))
        #
        #     # Load video labels
        #     database, comment, previous_data = load_database(selected_video)
        #
        #     # Load labels on checkboxes, score and comment
        #     window.fill(database.loc[selected_video, label_dict['key']].to_dict())
        #     # window['region-score'].update(str(database.loc[selected_video, 'score']))
        #     window['comment'].update(value=comment)


        # Manually save changes
        if event == 'save':
            for key in label_dict['key']:  # update database
                database[key] = int(values[key])

            comment = values['comment']
            save_database(selected_video, database, comment)
            # get_score(database)
            # database.loc[selected_video, 'comment'] = values['comment']  # save comment

            # database_filepath = 'labels/' + os.path.splitext(selected_video)[0] + '.csv'  # save labels
            # database.to_csv(database_filepath, sep=';')  # write to csv

            # comment_filepath = 'labels/' + os.path.splitext(selected_video)[0] + '.txt'  # save comment
            #
            # with open(comment_filepath, 'w') as f:
            #     f.write(comment)

        # Show motion detection
        if event == 'motion':
            minimo = np.min(dset_data[selected_video].bscan)
            maximo = np.max(dset_data[selected_video].bscan)
            video_array = (255 * (dset_data[selected_video].bscan - minimo) / (maximo - minimo)).astype('uint8')
            motion_detection(video_array)
            sg.popup('Video motion', image='velocity.png')

        # Go to test mode
        if event == 'test':
            database, comment, previous_data = load_database(selected_video)
            if previous_data == True:
                test_mode(database)
            else:
                sg.popup('There are no previously saved labels to compare with')

        # Pleura segmentation
        if event == 'pleura_detect':
            if pleura_mode == 'on':
                pleura_mode = 'off'
            else:
                pleura_mode = 'on'

        # Similar image finder
        if event == 'similar':
            input_frame = dset_data[selected_video].frames[:, :, n_frame]
            similar_indices = find_similar(input_frame, encoder, database_encoded)

            similar_mode(similar_indices, database_display, database_filenames)

        #### Update displayed image (if video selected)
        if not selected_video == None:
            n_frame = i % len(video_data[selected_video])
            i += 1  # Advance one frame

            # Generate pleura square mask
            dset_square_frame = dset_data[selected_video].bscan[:, :, n_frame]  # select frame from Dataset object
            dset_frame = dset_data[selected_video].frames[:, :, n_frame]  # select frame from Dataset object

            display_shape = dset_frame.shape  # get display shape

            input_square_img = load_img(dset_square_frame)  # preprocess image for model input

            square_mask = predict(input_square_img, pleura_square_model)  # predict mask
            square_mask_resized = cv2.resize(src=square_mask,
                                             dsize=(display_shape[1], display_shape[0]))  # resize mask to display shape
            norm_square_mask_img = np.uint8(square_mask_resized * 255)

            ## FRAME QUALITY ANALYSIS
            ## Analyze frame quality parameters
            frame_analysis = analyze_frame(dset_frame, square_mask)

            ## Check image quality
            if is_valid(frame_analysis[0], frame_analysis[1]):
                window['quality'].update(0)  # Set green (valid)
            else:
                window['quality'].update(1)  # Set red (not valid)

            # Display original image or mask
            if pleura_mode == 'off':
                ## Display original image
                window['display'].update(data=video_data[selected_video][n_frame])

            if pleura_mode == 'on':
                # Generate mask
                input_img = load_img(dset_frame)

                mask = predict(input_img, pleura_model)
                mask_resized = cv2.resize(src=mask,
                                          dsize=(display_shape[1], display_shape[0]))  # resize mask to display shape
                norm_mask_img = np.uint8(mask_resized * 255)

                ## Display blended mask
                norm_input_img = dset_frame / 48 + 1
                composite_img = blend_mask(np.uint8(norm_input_img * 255), norm_mask_img)
                window['display'].update(data=composite_img)

                ## Display only mask
                # window['display'].update(data=get_img_data(norm_mask_img))
