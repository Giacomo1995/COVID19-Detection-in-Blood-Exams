# Imports
import numpy as np


def get_header(dataset):
    """
    It gets the dataset, then saves and returns its header (i.e. the set of features).
    """

    header = np.array(dataset.columns)
    np.save('header.npy', header)  # Save header

    return header


def filter_dataset(dataset, header=[], header_name='header.npy'):
    """
    It filters the dataset by projecting the indicated features (header) through parameter value or file name.
    """

    # If the header parameter is left empty, load it by header_name
    if header is None:
        header = []
    if header == []:
        header = np.load(header_name, allow_pickle=True)

    dataset = dataset[header]  # Project the features on the dataset
    
    return dataset
