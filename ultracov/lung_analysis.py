
import numpy as np
from ultracov.file_functions import BinFile, Dataset  #From Jorge to read bin files
from skimage.transform import resize
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import tensorflow as tf


def bin_to_key(dset, n_clusters=5):
    minimo = np.min(dset.bscan)
    maximo = np.max(dset.bscan)
    data = (255*(dset.bscan-minimo)/(maximo-minimo)).astype('uint8')
    # Returns data array of shape (n_frames, img_dim)
    Nx, Ny, Nt = np.shape(data)
    img = resize(data, (Ny, Ny))  # Making the image square
    Nx, Ny, Nt = np.shape(img)
    #print('Size= ', Nx, Ny, Nt)
    # Reshape the image to use it as input of the K-means algorithm
    data1D = np.transpose(img.reshape((Nx*Ny,-1)))  # (nsamples, nfeatures)
    # Clusters --> Finding the image closest to the Centroid
    kmeans = KMeans(n_clusters)
    kmeans.fit(data1D)
    labels = kmeans.labels_
    centers = np.array(kmeans.cluster_centers_)
    closest, distances = vq(centers, data1D)

    key_frames = []
    for n in range(n_clusters):
        key_frames.append(img[:, :, closest[n]])

    return key_frames


def predict_analysis(key_frames, model):
    prediction_list = []
    for image in key_frames:
        prepared_image = np.expand_dims(np.expand_dims(image, axis=0), axis=3)
        x = model.predict(prepared_image)
        prediction_list.append(x[0])

    return np.array(prediction_list)


def get_analysis_labels(key_frames, orientation_model, region_model, score_model):
    predicted_orientation = predict_analysis(key_frames, orientation_model)
    predicted_orientation = round(np.average(predicted_orientation))

    predicted_region = predict_analysis(key_frames, region_model)
    predicted_region = round(np.argmax(predicted_region, axis=1).mean())

    predicted_score = predict_analysis(key_frames, score_model)
    predicted_score = round(np.argmax(predicted_score, axis=1).max())

    if predicted_orientation == 0:
        predicted_orientation_label = 'longitudinal'
    else:
        predicted_orientation_label = 'transversal'

    if predicted_region < 6:
        predicted_region_label = 'L'
    else:
        predicted_region_label = 'R'
    predicted_region_label += str((predicted_region % 6) + 1)

    return predicted_orientation_label, predicted_region_label, predicted_score


if __name__ == "__main__":
    filename = r"C:\Principal\ultracov\videos\PUERTADEHIERRO\4\4_L1_0.BIN"
    bfile = BinFile(filename)
    dset = Dataset(bfile)
    key_frames = bin_to_key(dset)

    # load model and make prediction
    orientation_model_path = r"C:\ultracov-project\ultracov\orientation_model.h5"
    region_model_path = r"C:\ultracov-project\ultracov\region_model.h5"
    score_model_path = r"C:\ultracov-project\ultracov\score_model.h5"

    orientation_model = tf.keras.models.load_model(orientation_model_path)
    region_model = tf.keras.models.load_model(region_model_path)
    score_model = tf.keras.models.load_model(score_model_path)

    print('Analysis labels: ', get_analysis_labels(key_frames, orientation_model, region_model, score_model))


