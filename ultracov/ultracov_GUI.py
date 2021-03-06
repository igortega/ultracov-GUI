# -*- coding: utf-8 -*-
"""
Interfaz para etiquetar exploraciones y generar base de datos

"""

import PySimpleGUI as sg
import cv2
import os
import numpy as np

import ultracov
from ultracov.motion_detection import motion_detection
from ultracov.mask_generator import load_model, prepare_for_predict, predict, blend_mask
from ultracov.similarity import init_similarity, find_similar
from ultracov.frame_quality import analyze_frame, is_valid
from ultracov.video_player import get_img_data, load_video_data
from ultracov.lung_analysis import bin_to_key, get_analysis_labels


def main():
    ##  AUXILIAR FUNCTIONS

    def get_score(database):
        """ Calculate score of a labelled region and updates database
            INPUT:
                - database: DataFrame with video labels """

        score3 = bool(sum([int(database[key]) for key in ['effusion', 'consolidation -', 'consolidation +']]))
        score2 = bool(sum([int(database[key]) for key in ['confluent +']]))
        score1 = bool(sum([int(database[key]) for key in ['blines', 'confluent -']]))
        score0 = bool(sum([int(database[key]) for key in ['alines']]))
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
        # label_list = [fname for fname in os.listdir(selected_directory) if os.path.splitext(fname)[-1] == '.csv']

        database_filename = os.path.splitext(selected_video)[0] + '.csv'
        comment_filename = os.path.splitext(selected_video)[0] + '.txt'

        database_filepath = os.path.join(selected_directory, database_filename)
        comment_filepath = os.path.join(selected_directory, comment_filename)

        keys = label_dict['key']
        # Previously saved label database
        if os.path.exists(database_filepath):
            with open(database_filepath) as f:
                labels_data = f.readlines()
            values = labels_data[1].split('\n')[0].split(';')
            values = [int(v) for v in values]

            if os.path.exists(comment_filepath):
                with open(comment_filepath) as f:
                    try:
                        comment = f.readlines()[0]
                    except:
                        comment = ''
            else:
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
        database_filename = os.path.splitext(selected_video)[0] + '.csv'
        comment_filename = os.path.splitext(selected_video)[0] + '.txt'

        database_filepath = os.path.join(selected_directory, database_filename)
        comment_filepath = os.path.join(selected_directory, comment_filename)

        keys = label_dict['key']
        with open(database_filepath, 'w') as f:
            keys_line = ';'.join(keys)
            values = list(database.values())
            values_line = ';'.join(list(map(str, values)))
            f.writelines([keys_line + '\n', values_line])
            f.close()

        with open(comment_filepath, 'w') as f:
            f.write(comment)

    #  INITIALIZE
    'assume all required files are present'
    # # Get required files
    # if not os.path.exists('required_files.zip'):
    #     get_files()

    # Read list of .bin videos in selected directory
    # selected_directory = sg.popup_get_folder('Select video directory')

    # video_list = [fname for fname in os.listdir(selected_directory) if os.path.splitext(fname)[-1] in ('.bin', '.BIN')]

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
                          'confluent -',
                          'confluent +',
                          'effusion',
                          'consolidation -',
                          'consolidation +']}

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
    # here = os.path.split(ultracov.__file__)[0]
    print(ultracov.here)
    sector_model_path = os.path.join(ultracov.here, 'pleura', 'pleura_model.h5')
    square_model_path = os.path.join(ultracov.here, 'pleura', 'pleura_square_model.h5')
    orientation_model_path = os.path.join(ultracov.here, 'orientation_model.h5')
    region_model_path = os.path.join(ultracov.here, 'region_model.h5')
    score_model_path = os.path.join(ultracov.here, 'score_model.h5')

    pleura_sector_model = load_model(sector_model_path)
    pleura_square_model = load_model(square_model_path)
    orientation_model = load_model(orientation_model_path)
    region_model = load_model(region_model_path)
    score_model = load_model(score_model_path)

    encoder, database_display, database_encoded, database_filenames = init_similarity()

    # LAYOUT ELEMENTS
    select_directory_row = [sg.Text(text='No directory selected',
                                    background_color='#ffffff',
                                    text_color='#000000',
                                    enable_events=True,
                                    key='directory_text'),
                            sg.Button(button_text='Select directory',
                                      enable_events=True,
                                      key='directory_button')]

    videos_listbox = [sg.Listbox(values=[],
                                 key='listbox',
                                 size=(30, 20),
                                 enable_events=True,
                                 font=('helvetica', 15),
                                 auto_size_text=True)]

    image_display = [sg.Image(key='display',
                              filename=os.path.join(ultracov.here, 'logo.png'),
                              background_color='#eeeeee')]

    quality_bar = [sg.T('Image quality'),
                   sg.ProgressBar(1,
                                  size=(15, 10),
                                  key='quality',
                                  bar_color=['red', 'green'])]

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

    # analysis_buttons2 = [sg.Button('Lung analysis',
    #                                key='analysis',
    #                                disabled=True)]

    # score_display = [sg.Text('Region score:',
    #                          auto_size_text=True,
    #                          font=('helvetica', 15)),
    #                  sg.Text(str('-'),
    #                          key='region-score',
    #                          auto_size_text=True,
    #                          font=('helvetica', 15))]

    analysis_frame_layout = [[sg.Text(text='Region'),
                              sg.Text(text='------',
                                      auto_size_text=True,
                                      key='analysis-region')],
                             [sg.Text(text='Score'),
                              sg.Text(text='-------',
                                      auto_size_text=True,
                                      key='analysis-score')],
                             [sg.Text(text='Orientation'),
                              sg.Text(text='-----------------',
                                      auto_size_text=True,
                                      key='analysis-orientation')]]

    analysis_frame = [sg.Frame(title='Analysis',
                               layout=analysis_frame_layout)]

    label_frame_layout = [[sg.Text(text='Region'),
                           sg.Text(text='------',
                                   auto_size_text=True,
                                   key='label-region')],
                          [sg.Text(text='Score'),
                           sg.Text(text='------',
                                   auto_size_text=True,
                                   key='label-score')]]

    label_frame_layout += [[sg.Checkbox(text=label_dict['name'][i],
                                        key=label_dict['key'][i],
                                        enable_events=True,
                                        disabled=True,
                                        checkbox_color=label_color[i],
                                        text_color='#000000',
                                        background_color='#dddddd',
                                        font=('helvetica', 20),
                                        auto_size_text=True)] for i in range(len(label_dict['name']))]

    # label_checkboxes_frame = [sg.Frame(title='Labels',
    #                                    layout=label_frame_layout)]

    label_frame = [sg.Frame(title='Labels',
                            layout=label_frame_layout)]

    comment_box = [sg.Input(key='comment',
                            size=(60, 60),
                            font=('helvetica', 10),
                            enable_events=True)]

    # LAYOUT COLUMNS
    video_selection_column = sg.Column([select_directory_row,
                                        videos_listbox])

    video_display_column = sg.Column([quality_bar,
                                      image_display,
                                      analysis_buttons],
                                     element_justification='center',
                                     background_color='#eeeeee')

    label_column = sg.Column([analysis_frame,
                              label_frame,
                              comment_box])

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

        if event == 'directory_button':
            selected_directory = sg.popup_get_folder('Select directory')
            video_list = [fname for fname in os.listdir(selected_directory) if
                          os.path.splitext(fname)[-1] in ('.bin', '.BIN')]

            window['directory_text'].update(value=selected_directory)
            window['listbox'].update(values=video_list)

            selected_video = None
            database = None
            window['display'].update(filename=os.path.join(ultracov.here, 'logo.png'))
            window['label-region'].update('-')
            window['label-score'].update('-')

            window['analysis-region'].update('-')
            window['analysis-score'].update('-')
            window['analysis-orientation'].update('-')

        # Activate/deactivate label checkboxes and buttons
        if selected_video is not None:
            for key in label_dict['key']:
                window[key].update(disabled=False)
            window['motion'].update(disabled=False)
            window['pleura_detect'].update(disabled=False)
            window['similar'].update(disabled=False)
            window['test'].update(disabled=False)

        else:
            for key in label_dict['key']:
                window[key].update(disabled=True)
            window['motion'].update(disabled=True)
            window['pleura_detect'].update(disabled=True)
            window['similar'].update(disabled=True)
            window['test'].update(disabled=True)

        # Tick checkbox
        if event in label_dict['key']:
            # database.loc[selected_video, event] = int(values[event])
            database[event] = int(values[event])  # update database
            # Update score
            window['label-score'].update(str(get_score(database)))

        # Select new video
        if event == 'listbox':
            if database is not None:
                save_database(selected_video, database, comment)  # save previous video's labels

            selected_video = values['listbox'][0]
            n_frame = 0

            # Load video data
            if selected_video not in video_data.keys():
                video_data[selected_video], bfile_data[selected_video], dset_data[selected_video] = load_video_data(
                    os.path.join(selected_directory, selected_video))

            # Load video labels
            database, comment, previous_data = load_database(selected_video)

            # Perform analysis
            key_frames = bin_to_key(dset_data[selected_video])
            predicted_orientation_label, predicted_region_label, predicted_score = get_analysis_labels(key_frames,
                                                                                                       orientation_model,
                                                                                                       region_model,
                                                                                                       score_model)
            # Load labels on checkboxes, score and comment
            window.fill(database)
            window['label-score'].update(str(get_score(database)))
            selected_video_region = selected_video.split('_')[1]
            window['label-region'].update(value=selected_video_region)
            window['comment'].update(value=comment)

            window['analysis-score'].update(str(predicted_score))
            window['analysis-region'].update(predicted_region_label)
            window['analysis-orientation'].update(predicted_orientation_label)

        if event == 'comment':
            comment = values[event]  # update comment

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
            save_database(selected_video, database, comment)
            database, comment, previous_data = load_database(selected_video)
            if previous_data:
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

        # Update displayed image (if video selected)
        if selected_video is not None:
            n_frame = i % len(video_data[selected_video])
            i += 1  # Advance one frame

            # Generate pleura square mask
            dset_square_frame = dset_data[selected_video].bscan[:, :, n_frame]  # select frame from Dataset object
            dset_frame = dset_data[selected_video].frames[:, :, n_frame]  # select frame from Dataset object

            display_shape = dset_frame.shape  # get display shape

            input_square_img = prepare_for_predict(dset_square_frame)  # preprocess image for model input

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
                input_img = prepare_for_predict(dset_frame)

                mask = predict(input_img, pleura_sector_model)
                # mask_resized = cv2.resize(src=mask,
                #                           dsize=(display_shape[1], display_shape[0]))  # resize mask to display shape
                # norm_mask_img = np.uint8(mask_resized * 255)

                ## Display blended mask
                composite_img = blend_mask(dset_frame, mask)
                window['display'].update(data=composite_img)

                ## Display only mask
                # window['display'].update(data=get_img_data(norm_mask_img))


if __name__ == "__main__":
    main()
