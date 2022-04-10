# Imports
import numpy as np


def get_header(training_set):
    """
    It gets the training set, then saves and returns its header (i.e. the set of features).
    """

    header = np.array(training_set.columns)
    np.save('header.npy', header)  # Save header

    return header


def filter_test_set(test_set, header=[], header_name='header.npy'):
    """
    It filters the test set by projecting the indicated features (header) through parameter value or file name.
    """

    # If the header parameter is left empty, load it by header_name
    if header is None:
        header = []
    if header == []:
        header = np.load(header_name, allow_pickle=True)

    test_set = test_set[header]  # Project the features on the test set
    
    return test_set
