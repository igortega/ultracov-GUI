# -*- coding: utf-8 -*-
"""
Created on 16/02/2021
@author: Jorge Camacho
"""

import struct
import numpy as np
import array
from matplotlib.pyplot import *
import scipy.signal as sig
import scipy.ndimage as ndi
from time import time
from numba import jit


class BinFile:
    # Objeto con el mismo contenido que el archivo .bin
    # todo: Revisar que los canales virtuales se manejan bien. Por ahora sólo estoy seguro de que se lee
    #       bien el primero, que es el que usamos en ultracov.

    # Variables de objeto
    filepath: str
    version: str
    date: str
    dll_version: str
    firmware_boot_version: str
    firmware_ethernet_version: str
    firmware_uci_sw_version: str
    firmware_uci_hw_version: str
    firmware_base_version: str
    firmware_module_version: str
    n_total_channels: int
    n_active_channels: int
    n_multiplexed_channels: int
    trigger_lines_number: int
    auxiliar_lines_number: int
    trigger_scan_resolution: float
    trigger_scan_length: float
    auxiliar_scan_resolution: float
    auxiliar_scan_length: float
    trigger_source: int
    prf_time_images: int
    log_dynamic_range: float
    virtual_channels_number: int
    start_range_mm: float
    start_range_us: float
    end_range_mm: float
    end_range_us: float
    n_asca: int
    n_samples: int
    sampling_frequency: float
    reduction_factor: int
    gain: float
    x_entry_points: float
    z_entry_points: float
    angle_entry_points: float
    scan_data: int
    sw_time_stamp: float

    def __init__(self, filepath):

        self.filepath = filepath

        # Abro el archivo
        try:
            f = open(self.filepath, "rb")
        except:
            print("ERROR: Cannot open .bin file")

        # Veo si es little o big endian
        e = '<'  # Supongo little endian
        masc, = struct.unpack(e + 'H', f.read(2))
        if masc == 4660:  # El formato es little endian
            pass
        elif masc == 13330:  # El formato es big endian
            e = '>'  # Cambio el formato a big endian
        else:
            raise Exception("ERROR: Control character not valid")

        # ---> Configuration File  Version
        self.version = ReadBINFileField_STRING(f, e)

        # ---> Date, Time
        self.date = ReadBINFileField_STRING(f, e)

        # ---> DLL Version
        self.dll_version = ReadBINFileField_STRING(f, e)

        # ---> Firmware Boot Version
        self.firmware_boot_version = ReadBINFileField_STRING(f, e, read=False)

        # ---> Firmware Ethernet Version
        self.firmware_ethernet_version = ReadBINFileField_STRING(f, e)

        # ---> Firmware UCI SW Version
        self.firmware_uci_sw_version = ReadBINFileField_STRING(f, e)

        # ---> Firmware UCI HW Version
        self.firmware_uci_hw_version = ReadBINFileField_STRING(f, e)

        # ---> Firmware BASE Version
        self.firmware_base_version = ReadBINFileField_STRING(f, e)

        # ---> Firmware MODULE Version
        self.firmware_module_version = ReadBINFileField_STRING(f, e)

        # ---> Total Channels
        self.n_total_channels = ReadBINFileField_INT(f, e)

        # ---> Active Channels
        self.n_active_channels = ReadBINFileField_INT(f, e)

        # ---> Multiplexed Channels
        self.n_multiplexed_channels = ReadBINFileField_INT(f, e)

        # // _____________________________________________________________________________
        # // ------ S C A N   C O N F I G
        # // =============================================================================

        # ---> Trigger Lines Number
        self.trigger_lines_number = ReadBINFileField_INT(f, e)

        # ---> Auxiliar Lines Number
        self.auxiliar_lines_number = ReadBINFileField_INT(f, e)

        # ---> (mm) Trigger Scan Resolution
        self.trigger_scan_resolution = ReadBINFileField_FLOAT(f, e)

        # ---> (mm) Trigger Scan Length
        self.trigger_scan_length = ReadBINFileField_FLOAT(f, e)

        # ---> (mm) Auxiliar Scan Resolution
        self.auxiliar_scan_resolution = ReadBINFileField_FLOAT(f, e)

        # ---> (mm) Auxiliar Scan Length
        self.auxiliar_scan_length = ReadBINFileField_FLOAT(f, e)

        # ---> Trigger Source
        self.trigger_source = ReadBINFileField_INT(f, e)

        if self.version != '0000.0000.0000.0001':
            # ---> Configured PRF between images
            self.prf_time_images = ReadBINFileField_INT(f, e)

        # // _____________________________________________________________________________
        # // ------ V I R T U A L   C H A N N E L   D A T A
        # // =============================================================================

        # ---> Virtual Channels Number
        self.virtual_channels_number = ReadBINFileField_INT(f, e)

        # Defino los arrays
        self.n_samples = np.zeros(self.virtual_channels_number, dtype='int')
        self.n_ascan = np.zeros(self.virtual_channels_number, dtype='int')
        self.start_range_mm = np.zeros(self.virtual_channels_number, dtype='float')
        self.start_range_us = np.zeros(self.virtual_channels_number, dtype='float')
        self.end_range_mm = np.zeros(self.virtual_channels_number, dtype='float')
        self.end_range_us = np.zeros(self.virtual_channels_number, dtype='float')
        self.sampling_frequency = np.zeros(self.virtual_channels_number, dtype='float')
        self.reduction_factor = np.zeros(self.virtual_channels_number, dtype='int')
        self.gain = np.zeros(self.virtual_channels_number, dtype='float')
        self.prf_time_lines = np.zeros(self.virtual_channels_number, dtype='float')
        self.log_dynamic_range = np.zeros(self.virtual_channels_number, dtype='float')

        self.x_entry_points = []
        self.z_entry_points = []
        self.angle_entry_points = []
        self.scan_data = []
        self.sw_time_stamp = []

        for i in range(self.virtual_channels_number):

            # // _____________________________________________________________________________
            # // ---------- AQUISITION DATA CONFIG -------------------------------------------
            # // -----------------------------------------------------------------------------

            # ---> A - Scan Samples Number
            self.n_samples[i] = ReadBINFileField_INT(f, e)

            # ---> Scan Image Lines(A - Scan)
            self.n_ascan[i] = ReadBINFileField_INT(f, e)

            # ---> (mm) Start Range
            self.start_range_mm[i] = ReadBINFileField_FLOAT(f, e)

            # ---> (us) Start Range
            self.start_range_us[i] = ReadBINFileField_FLOAT(f, e)

            # ---> (mm) End Range
            self.end_range_mm[i] = ReadBINFileField_FLOAT(f, e)

            # ---> (us) End Range
            self.end_range_us[i] = ReadBINFileField_FLOAT(f, e)

            # ---> (MHz) Sampling Frequency
            self.sampling_frequency[i] = ReadBINFileField_FLOAT(f, e)

            # ---> Reduction Factor
            self.reduction_factor[i] = ReadBINFileField_INT(f, e)

            # ---> (dB) Gain
            self.gain[i] = ReadBINFileField_FLOAT(f, e)

            if self.version != '0000.0000.0000.0001':
                # ---> PRF time between lines
                self.prf_time_lines[i] = ReadBINFileField_FLOAT(f, e)

                # ---> Image dynamic range in dB
                self.log_dynamic_range[i] = ReadBINFileField_FLOAT(f, e)

            # ---> (mm) Entry Points X Coordinates
            buffer_float = ReadBINFileField_FLOAT(f, e)
            self.x_entry_points.append(np.array(buffer_float))

            # ---> (mm) Entry Points Z Coordinates
            buffer_float = ReadBINFileField_FLOAT(f, e)
            self.z_entry_points.append(np.array(buffer_float))

            # ---> (º) Entry Points Angles
            buffer_float = ReadBINFileField_FLOAT(f, e)
            self.angle_entry_points.append(np.array(buffer_float))

            # ---> Data
            buffer_short = ReadBINFileField_SHORT(f, e)
            self.scan_data.append(np.array(buffer_short))
            del buffer_short

            if self.version != '0000.0000.0000.0001':
                # ---> Sw time-stamp
                buffer_double = ReadBINFileField_DOUBLE(f, e)
                self.sw_time_stamp.append(np.array(buffer_double))

        f.close()

        print('File version:    ' + self.version)
        print('File date:       ' + self.date)
        print('Total Channels:  ' + str(self.n_total_channels))
        print('Active Channels: ' + str(self.n_active_channels))
        print('Multiplexed Channels: ' + str(self.n_active_channels))
        print('Dll version:              ' + self.dll_version)
        print('Firmware boot version     ' + self.firmware_boot_version)
        print('Firmware ethernet version ' + self.firmware_ethernet_version)
        print('Firmware UCI SW version   ' + self.firmware_uci_sw_version)
        print('Firmware UCI HW version   ' + self.firmware_uci_hw_version)
        print('Firmware BASE version     ' + self.firmware_base_version)
        print('Firmware MODULE version   ' + self.firmware_module_version)


class Dataset:
    # Datos en formato cubo con variables de rejilla para pintar
    filename: str  # Nombre del archivo
    nr: int  # Número de muestras de cada Ascan
    nlin: int  # Número de líneas de la imagen
    nimg: int  # Número de imágenes en el video
    x_ini: float  # Coordenadas X de los puntos iniciales de los A-Scan
    z_ini: float  # Coordenadas Z de los puntos iniciales de los A-Scan
    r: float  # Vector de distancias de los puntos de los A-Scan
    x: float  # Matriz de coordenadas X de los puntos
    z: float  # Matriz de coordenadas Y de los puntos
    med_kernel: int  # Tamaño del filtro de mediana móvil
    angles: float  # Vector de ángulos del barrido
    px2mm: float  # Conversión de píxeles a mm
    frame_ny: int  # Nº de píxeles verticales
    frame_nx: int  # Nº de píxeles horizontales
    x_px: int  # Coordenada X de las muestras en píxeles
    y_px: int  # Coordenada Y de las muestras en píxeles
    xl: int  # Coordenada x del inicio de la primera línea (en px)
    xr: int  # Coordenada x del inicio de la última línea (en px)
    yl: int  # Coordenada y del fin de la primera línea (en px)
    yr: int  # Coordenada y del fin de la última línea (en px)
    cx: int  # Coordenada x del centro del barrido (en px)
    cy: int  # Coordenada y del centro del barrido (en px)
    frames_val: bool  # Máscara de píxeles válidos
    frames: float  # Imágenes en píxeles
    # todo: este filtro lo va a hacer Ricardo en C, que es más rápido
    fps: int  # Frames por segundo del video
    dB_range: float  # Rango dinámico de la imagen

    def __init__(self, binFile: BinFile, avg_kernel=7, med_kernel=3, resample=1000, new_db_range=0, data=True):

        # Si resample > 0 se reduce el número de muestras en la dirección de propagación a ese valor (aproximadamente)
        # Es para evitar trabajar con imágenes sobre-muestreadas, que enlentecen el procesamiento

        if resample > 0 and binFile.n_samples[0] > resample:

            resample_factor = np.ceil(binFile.n_samples[0] / resample).astype('int')
            self.nr = np.ceil(binFile.n_samples[0] / resample_factor).astype('int')

        else:
            self.nr = binFile.n_samples[0]
            resample_factor = 1

        self.filename = binFile.filepath

        self.med_kernel = med_kernel  # Tamaño del filtro de mediana móvil para reducir speckle
        # Para desactivar el filtro poner med_kernel = 0
        self.avg_kernel = avg_kernel  # Tamaño del filtro de media móvil (0 desactiva)

        self.nlin = binFile.n_ascan[0]
        self.nimg = binFile.trigger_lines_number
        self.x_ini = binFile.x_entry_points[0][:]
        self.z_ini = binFile.z_entry_points[0][:]
        self.r = np.arange(binFile.start_range_mm, binFile.end_range_mm,
                           (binFile.end_range_mm - binFile.start_range_mm) / self.nr)
        self.angles = binFile.angle_entry_points[0][:]

        # Matrices de coordenadas de los píxeles
        self.x = np.zeros([self.nr, self.nlin], dtype='float')
        self.z = np.zeros([self.nr, self.nlin], dtype='float')
        for i in range(self.nlin):
            self.x[:, i] = self.r * np.sin(binFile.angle_entry_points[0][i]) + self.x_ini[i]
            self.z[:, i] = - self.r * np.cos(binFile.angle_entry_points[0][i]) + self.z_ini[i]
        # Desplazo las coordenadas horizontales para tener el 0 en el centro de la imagen. A veces no está en el centro
        self.x = self.x - ((np.max(self.x) - np.min(self.x)) / 2 + np.min(self.x))

        if data:
            # Imagenes del video
            self.bscan = np.zeros([self.nr, self.nlin, self.nimg], dtype='float')
            ns = binFile.n_samples[0] * self.nlin  # número de muestras de una imagen original

            for i in range(self.nimg):
                ix_frame = np.arange(i * ns, (i + 1) * ns, dtype='int')
                self.bscan[:, :, i] = (np.reshape(binFile.scan_data[0][ix_frame],
                                                  [self.nlin, binFile.n_samples[0]]).T / 10)[0::resample_factor,
                                      :].copy()

                # Si está activado el filtro de mediana lo aplico
                if self.med_kernel > 0:
                    self.bscan[:, :, i] = ndi.median_filter(self.bscan[:, :, i], size=self.med_kernel)

                # Si está activado el filtro de media lo aplico
                if self.avg_kernel > 0:
                    self.bscan[:, :, i] = ndi.uniform_filter(self.bscan[:, :, i], size=self.avg_kernel)
                    # self.bscan[:, :, i] = ndi.uniform_filter1d(self.bscan[:, :, i], size=self.avg_kernel, axis=0)
                    # self.bscan[:, :, i] = ndi.maximum_filter1d(self.bscan[:, :, i], size=self.avg_kernel, axis=0)

            if binFile.version == '0000.0000.0000.0001':

                # Versión antigua del archivo
                self.fps = 20  # FPS por defecto en archivos antiguos
                self.dB_range = 45  # Rango dinámico por defecto en archivos antiguos

            else:

                self.fps = np.round(1 / binFile.prf_time_images * 1000)  # FPS teóricos del video
                self.dB_range = - binFile.log_dynamic_range[0]  # Rango dinámico de la imagen

            if new_db_range > 0:
                self.bscan[self.bscan < -new_db_range] = -new_db_range
                self.dB_range = np.array(new_db_range)

    # @jit(nopython=True, parallel=False, cache=True)
    def ScanConvert(self, InterpType='bilinear'):
        # Convierto las imágenes angulares a frames rectangulares.
        # Por ahora el algoritmo es vecino más próximo

        self.px2mm = (self.r[-1] - self.r[0]) / self.nr  # Calculo el factor de escala en vertical

        # Calculo el tamaño de la imagen y las rejillas XY
        self.frame_ny = np.round((np.max(self.z) - np.min(self.z)) / self.px2mm).astype(
            int)  # número de puntos en vertical
        self.frame_nx = np.round((np.max(self.x) - np.min(self.x)) / self.px2mm).astype(
            int)  # número de puntos en horizontal

        self.x_px = np.round((self.x - np.min(self.x)) / self.px2mm).astype(int)
        self.y_px = -np.round((self.z - np.max(self.z)) / self.px2mm).astype(int)

        # Calculo el centro del barrido en píxeles
        xl = np.round((self.x[0, 0] - np.min(self.x)) / self.px2mm)
        xr = np.round((self.x[0, -1] - np.min(self.x)) / self.px2mm)
        yl = np.round(-self.z[-1, 0] / self.px2mm)
        yr = np.round(-self.z[-1, -1] / self.px2mm)
        cx = (xl + xr) / 2  # Coordenada x del centro del barrido
        cy = -(yl * cx) / xl + yl  # Coordenada y del centro del barrido
        self.xl = xl
        self.xr = xr
        self.yl = yl
        self.yr = yr
        self.cx = cx
        self.cy = cy

        # Calculo el rango inicial y final en píxeles, y los pasos radiales y angulares
        ri = np.sqrt((cx - xl) ** 2 + cy ** 2)
        rf = np.sqrt(cx ** 2 + (cy - yl) ** 2)
        dr = (self.r[1] - self.r[0]) / self.px2mm
        da = (self.angles[1] - self.angles[0])

        if InterpType == 'nearest':
            # Calculo la matriz de índices para el interpolador
            ix = np.zeros((self.frame_ny, self.frame_nx, 2), dtype=int)
            self.frames_val = np.zeros((self.frame_ny, self.frame_nx), dtype=bool)
            ScanConverterIndexes(self.frame_ny, self.frame_nx, cx, cy, ri, rf, da, dr, self.angles, ix, self.frames_val,
                                 self.nlin, self.nr)

            # Aplico el interpolador
            self.frames = np.ones((self.frame_ny, self.frame_nx, self.nimg)) * (-self.dB_range)
            ScanConvert(self.nimg, self.frame_ny, self.frame_nx, self.frames_val, self.frames, self.bscan, ix)
        elif InterpType == 'bilinear':
            # Calculo las matrices de índices y pesos del interpolador
            ix = np.zeros((self.frame_ny, self.frame_nx, 2), dtype=int)
            w = np.zeros((self.frame_ny, self.frame_nx, 4), dtype=float)
            self.frames_val = np.zeros((self.frame_ny, self.frame_nx), dtype=bool)
            ScanConverterIndexes_Bilineal(self.frame_ny, self.frame_nx, cx, cy, ri, rf, da, dr, self.angles, ix,
                                          self.frames_val,
                                          self.nlin, self.nr, w)

            # Aplico el interpolador bilineal
            self.frames = np.ones((self.frame_ny, self.frame_nx, self.nimg), dtype='h') * (-self.dB_range.astype('h'))
            ScanConvert_Bilineal(self.nimg, self.frame_ny, self.frame_nx, self.frames_val, self.frames, self.bscan, ix,
                                 w)
        else:
            raise Exception('Tipo de interpolador desconocido')


@jit(nopython=True, parallel=False, cache=True)
def ScanConverterIndexes(frame_ny, frame_nx, cx, cy, ri, rf, da, dr, angles, ix, val, nlin, nr):
    # matrices de índices a las muestras y pixel válido
    for i in range(frame_ny):
        for j in range(frame_nx):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)  # Distancia al centro
            alfa = np.arcsin((j - cx) / d)  # Ángulo con respecto a la vertical

            if (d > ri) and (d < rf) and (alfa > angles[0]) and (alfa < angles[-1]):
                ix[i, j, 1] = np.round((alfa - angles[0]) / da)  # ángulo
                ix[i, j, 0] = np.round((d - ri) / dr)

                if (ix[i, j, 1] >= 0) and (ix[i, j, 1] < nlin) and (ix[i, j, 0] >= 0) and (ix[i, j, 0] < nr):
                    val[i, j] = True


@jit(nopython=True, parallel=False, cache=True)
def ScanConvert(nimg, frame_ny, frame_nx, val, frames, bscan, ix):
    for ix_frame in range(nimg):
        for i in range(frame_ny):
            for j in range(frame_nx):
                if val[i, j]:
                    frames[i, j, ix_frame] = bscan[ix[i, j, 0], ix[i, j, 1], ix_frame]


@jit(nopython=True, parallel=False, cache=True)
def ScanConverterIndexes_Bilineal(frame_ny, frame_nx, cx, cy, ri, rf, da, dr, angles, ix, val, nlin, nr, w):
    # matrices de índices a las muestras y pixel válido
    for i in range(frame_ny):
        for j in range(frame_nx):
            d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)  # Distancia al centro
            alfa = np.arcsin((j - cx) / d)  # Ángulo con respecto a la vertical

            if (d > ri) and (d < rf) and (alfa > angles[0]) and (alfa < angles[-1]):
                ixr = (d - ri) / dr
                ixth = (alfa - angles[0]) / da

                ix[i, j, 1] = np.floor(ixth)  # ángulo
                ix[i, j, 0] = np.floor(ixr)  # radio

                if (ix[i, j, 1] >= 0) and (ix[i, j, 1] < nlin - 1) and (ix[i, j, 0] >= 0) and (ix[i, j, 0] < nr - 1):
                    # Calculo las proporciones
                    dn = ixr - np.floor(ixr)
                    dn1 = np.ceil(ixr) - ixr
                    thm = ixth - np.floor(ixth)
                    thm1 = np.ceil(ixth) - ixth

                    # Calculo los pesos
                    w[i, j, 0] = dn1 * thm1
                    w[i, j, 1] = dn * thm1
                    w[i, j, 2] = dn1 * thm
                    w[i, j, 3] = dn * thm

                    val[i, j] = True


@jit(nopython=True, parallel=False, cache=True)
def ScanConvert_Bilineal(nimg, frame_ny, frame_nx, val, frames, bscan, ix, w):
    for ix_frame in range(nimg):
        for i in range(frame_ny):
            for j in range(frame_nx):
                if val[i, j]:
                    # Indices a la matriz
                    ixr1 = ix[i, j, 0]
                    ixr2 = ixr1 + 1
                    ixth1 = ix[i, j, 1]
                    ixth2 = ixth1 + 1

                    frames[i, j, ix_frame] = w[i, j, 0] * bscan[ixr1, ixth1, ix_frame] + \
                                             w[i, j, 1] * bscan[ixr2, ixth1, ix_frame] + \
                                             w[i, j, 2] * bscan[ixr1, ixth2, ix_frame] + \
                                             w[i, j, 3] * bscan[ixr2, ixth2, ix_frame]


def ReadBINFileField_STRING(file, e, read=True):
    data = 0
    error = 0
    APPBIN_iCAMPO = '['
    APPBIN_fCAMPO = ']'
    DATA_SIZE = 1

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_iCAMPO:
        raise Exception('Error Configuration File Format')

    data_size, = struct.unpack(e + 'I', file.read(4))

    if data_size != DATA_SIZE:
        raise Exception('Error Configuration File Format')

    n_data_file, = struct.unpack(e + 'I', file.read(4))  # nº de caracteres del campo

    if read:
        data = file.read(n_data_file).decode()  # leo y convierto a ascii
    else:
        data = file.read(n_data_file)
        data = 'Empty '

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_fCAMPO:
        raise Exception('Error Configuration File Format')

    return data[:-1]  # No sé porqué pero siempre hay un NULL '\x00' al final de los strings que tengo que quitar


def ReadBINFileField_INT(file, e):
    APPBIN_iCAMPO = '['
    APPBIN_fCAMPO = ']'
    DATA_SIZE = 4

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_iCAMPO:
        raise Exception('Error Configuration File Format')

    data_size, = struct.unpack(e + 'I', file.read(4))

    if data_size != DATA_SIZE:
        raise Exception('Error Configuration File Format')

    n_data_file, = struct.unpack(e + 'I', file.read(4))  # nº de bytes del campo

    if n_data_file > 1:
        data = array.array('i', file.read(n_data_file * DATA_SIZE))
        if e == '>':  # Big endian
            data.byteswap()
    else:
        data, = struct.unpack(e + 'i', file.read(DATA_SIZE))  # leo y convierto a int32

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_fCAMPO:
        raise Exception('Error Configuration File Format')

    return data


def ReadBINFileField_FLOAT(file, e):
    APPBIN_iCAMPO = '['
    APPBIN_fCAMPO = ']'
    DATA_SIZE = 4

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_iCAMPO:
        raise Exception('Error Configuration File Format')

    data_size, = struct.unpack(e + 'I', file.read(4))

    if data_size != DATA_SIZE:
        raise Exception('Error Configuration File Format')

    n_data_file, = struct.unpack(e + 'I', file.read(4))  # nº de bytes del campo

    if n_data_file > 1:
        data = array.array('f', file.read(n_data_file * DATA_SIZE))
        if e == '>':  # Big endian
            data.byteswap()
    else:
        data, = struct.unpack(e + 'f', file.read(DATA_SIZE))  # leo y convierto a int32

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_fCAMPO:
        raise Exception('Error Configuration File Format')

    return data


def ReadBINFileField_DOUBLE(file, e):
    APPBIN_iCAMPO = '['
    APPBIN_fCAMPO = ']'
    DATA_SIZE = 8

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_iCAMPO:
        raise Exception('Error Configuration File Format')

    data_size, = struct.unpack(e + 'I', file.read(4))

    if data_size != DATA_SIZE:
        raise Exception('Error Configuration File Format')

    n_data_file, = struct.unpack(e + 'I', file.read(4))  # nº de bytes del campo

    if n_data_file > 1:
        data = array.array('d', file.read(n_data_file * DATA_SIZE))
        if e == '>':  # Big endian
            data.byteswap()
    else:
        data, = struct.unpack(e + 'd', file.read(DATA_SIZE))  # leo y convierto a float64

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_fCAMPO:
        raise Exception('Error Configuration File Format')

    return data


def ReadBINFileField_SHORT(file, e):
    APPBIN_iCAMPO = '['
    APPBIN_fCAMPO = ']'
    DATA_SIZE = 2

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_iCAMPO:
        raise Exception('Error Configuration File Format')

    data_size, = struct.unpack(e + 'I', file.read(4))

    if data_size != DATA_SIZE:
        raise Exception('Error Configuration File Format')

    n_data_file, = struct.unpack(e + 'I', file.read(4))  # nº de bytes del campo

    if n_data_file > 1:
        data = array.array('h', file.read(n_data_file * DATA_SIZE))
        if e == '>':  # Big endian
            data.byteswap()
    else:
        data, = struct.unpack(e + 'h', file.read(DATA_SIZE))  # leo y convierto a int16

    if chr(struct.unpack(e + 'B', file.read(1))[0]) != APPBIN_fCAMPO:
        raise Exception('Error Configuration File Format')

    return data


if __name__ == "__main__":

    # Código de prueba: Carga un archivo .bin y lo pinta

    directory = r"C:\Principal\ultracov\videos\ensayos\4"  # Directorio de trabajo
    filename = directory + r'\4_L1_0.BIN'  # Nombre del archivo

    # Leo el archivo y genero un objeto 'BinFile' que contiene los datos crudos tal cual se leen del archivo
    bfile = BinFile(filename)

    # A partir del objeto 'BinFile' genero un objeto 'Dataset' que ya tiene los datos en formato 'cubo de datos'
    # y contiene las matrices de coordenadas de los píxeles para poder pintar
    dset = Dataset(bfile)
    dset.ScanConvert()

    # Pinto las imagenes en formato 'matriz' y con aspecto real
    figure(1).clf()
    ax1 = subplot(121)  # Imagen matriz
    ax2 = subplot(122)  # Imagen real

    for i in range(dset.nimg):
        # Frame que voy a pintar
        img = dset.bscan[:, :, i]

        # Pinto la imagen en formato 'matriz'
        ax1.cla()
        ax1.imshow(dset.frames[:, :, i], cmap='gray', interpolation='nearest', aspect='auto')
        ax1.set_title('Imagen en formato matricial')
        ax1.set_xlabel('Line number')
        ax1.set_ylabel('Sample number')
        ax1.axis('equal')

        # Pinto la imagen con su forma real
        ax2.cla()
        c = ax2.pcolormesh(dset.x, dset.z, img, cmap='gray')
        ax2.axis('equal')
        if i == 0:
            cb = colorbar(c, ax=ax2)
            cb.ax.set_title('dB')
        ax2.set_title('Imagen real')
        ax2.set_xlabel('x (mm)')
        ax2.set_ylabel('z (mm)')

        # Actualizo
        pause(0.001)

    # print(vars(bfile))