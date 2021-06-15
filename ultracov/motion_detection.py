# -*- coding: utf-8 -*-
"""Ultrasound video motion detection through optical flow algorithm

The function motion_detection() tracks most noticeable points of
a video inside a selected region. It calculates its motion
along x and z directions in pixel units.

If executed as a script, it runs an example.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ultracov.file_functions import BinFile, Dataset


def motion_detection(video_array, x1=0, x2=-1, z1=0, z2=-1):
    """ Detects motion in a video and makes a figure

    Parameters
    ----------
    video_array
        An array shaped (x length, z length, number of frames) normalized to (0,255)
    x1 : int
        Selected region corner coordinate
    x2 : int
        Selected region corner coordinate
    z1 : int
        Selected region corner coordinate
    z2 : int
        Selected region corner coordinate

    Returns
    -------
    It saves motion data in main directory as 'motion.dat'
    It creates and saves a plot figure in main directory as 'velocity.png'
    """

    # Numero de frames
    NIMG = np.shape(video_array)[2]

    # Parametros a la hora de localizar features (Se pueden optimizar) (Se buscan hasta 100 puntos en este caso)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parametros de lucas kanade optical flow 
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Se analiza la primera frame para obtener puntos significativos a seguir
    old_gray = video_array[:, :, 0]

    # La Mascara (basada en las coordenadas introducidas) permite seleccionar la región donde buscar puntos (features)    
    mask = np.zeros_like(old_gray)
    mask[z1:z2, x1:x2] = 255

    # Obtenemos los puntos
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

    dxv = []
    dzv = []

    for i in range(1, NIMG):  # Empezamos en 1 porque el frame 0 lo tomamos de referencia

        # Seleccionamos frame a analizar    
        frame_gray = video_array[:, :, i]

        # Saltar frames completamente blancos / negros
        if np.sum(frame_gray) in (255 * frame_gray.shape[0] * frame_gray.shape[1], 0):
            dxv.append([0.0])
            dzv.append([0.0])

        else:
            # Calculo del flujo optico
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if st is not None:
                # Seleccionamos los puntos buenos (los que hemos podido seguir)
                good_old = p0[st == 1]
                good_new = p1[st == 1]
                [dx, dz] = np.mean((good_new - good_old),
                                   axis=0)  # Calcula el desplazamiento medio de todos los puntos elegidos
                dxv.append([dx])  # Almacenar para analisis final
                dzv.append([dz])
            else:
                # Si no se encuentra correspondencia con el nuevo frame que permita seguir movimiento
                # Se vuelve a tomar esa frame como referencia y se buscan nuevos puntos a seguir 
                old_gray = video_array[:, :, i]
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
                continue

            # Actualizamos el frame y los puntos
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    np.savetxt('motion.dat', dxv, delimiter=',', fmt='%f')
    print("Motion Detection Completed")

    dxvf = savgol_filter(np.squeeze(dxv), 5, 3)  # Filtro (opcional)

    # Pinto las imagenes en formato 'matriz' y con aspecto real
    plt.ioff()
    plt.figure(2).clf()
    plt.plot(dxvf, label='Velocity')
    plt.xlabel('frame', fontsize=14)
    plt.ylabel('pixel/frame', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig('velocity.png')


if __name__ == "__main__":
    # execution example
    filename = r"C:\Principal\ultracov\videos\8_R1.BIN"

    # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
    bfile = BinFile(filename)

    # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
    # y contiene las matrices de coordenadas de los píxeles para poder pintar
    dset = Dataset(bfile)

    # La funcion del flujo optico usa imagenes con formato UINT8 (0..255)
    # Normalizamos los datos para adaptar a ese formato.
    minimo = np.min(dset.bscan)
    maximo = np.max(dset.bscan)
    video_array = (255 * (dset.bscan - minimo) / (maximo - minimo)).astype('uint8')

    motion_detection(video_array)
