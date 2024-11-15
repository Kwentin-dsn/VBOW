
import numpy as np
import glob
import time
import csv
from sklearn.cluster import MiniBatchKMeans, KMeans
import pandas as pd
import os


def getImgPaths(data_folder):
    """Returns filepaths of all files contained in the given folder as strings."""
    return np.sort(glob.glob(os.path.join(data_folder, '*')))


def toOneHot(indices, min_int, max_int):
    """Converts an enumerable of indices to a one-hot representation."""
    one_hot_length = max_int - min_int + 1
    eye = np.eye(one_hot_length)
    return eye[np.array(indices) - min_int]


def writePredictionsToCsv(predictions, out_path, label_strings):
    """Writes the predictions to a csv file.
    Assumes the predictions are ordered by test interval id."""
    with open(out_path, 'w') as outfile:
        csvwriter = csv.writer(
            outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL
        )
        # Write the header
        row_to_write = ['Id'] + [label for label in label_strings]
        csvwriter.writerow(row_to_write)
        # Write the rows using 18 digit precision
        for idx, prediction in enumerate(predictions):
            assert len(prediction) == len(label_strings)
            csvwriter.writerow(
                [str(idx + 1)] +
                ["%.18f" % p for p in prediction]
            )


def generateUniqueFilename(basename, file_ext):
    """Adds a timestamp to filenames for easier tracking of submissions, models, etc."""
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return basename + '_' + timestamp + '.' + file_ext


def createPath(path):
    """Creates a directory if it does not exist"""
    try:
        os.stat(path)
    except FileNotFoundError:
        os.makedirs(path)  # Create all necessary directories


def createCodebook(train_features, codebook_size=500):
    """
    Creates a codebook using KMeans clustering on the provided training features.
    
    Args:
        train_features (array-like): List or array of image features, each feature should be a 1D array.
        codebook_size (int): Number of clusters for the codebook.

    Returns:
        codebook: A fitted KMeans model representing the codebook.
    """
    # Ensure train_features is a 2D array for KMeans
    train_features = np.array(train_features)
    
    if train_features.ndim == 1:
        # Reshape to 2D if it's a 1D array of flattened descriptors
        train_features = train_features.reshape(-1, 1)
    elif train_features.ndim > 2:
        # Flatten each image's features into a single row (if not already flattened)
        train_features = train_features.reshape(len(train_features), -1)
    
    # Fit KMeans to create the codebook
    codebook = KMeans(n_clusters=codebook_size, random_state=42)
    codebook.fit(train_features)
    return codebook
    return codebook


def encodeImage(features, codebook):
    """Encodes one image given a visual BOW codebook"""
    # Find the minimal feature distance for all patches of the image
    visual_words = codebook.predict(features)
    word_occurrence = pd.DataFrame(visual_words, columns=['cnt'])['cnt'].value_counts()
    bovw_vector = np.zeros(codebook.n_clusters)
    for key in word_occurrence.keys():
        bovw_vector[key] = word_occurrence[key]
    bovw_feature = bovw_vector / np.linalg.norm(bovw_vector)
    return bovw_feature

