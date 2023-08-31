import numpy as np
import urllib.request
from io import StringIO

def load_dataset(dataset_name):
    '''
    Load real-world datasets. Processed datasets are from https://github.com/cmu-phil/example-causal-datasets/tree/main

    Parameters
    ----------
    dataset_name : str, ['sachs', 'boston_housing', 'airfoil']

    Returns
    -------
    data = np.array
    labels = list
    '''

    url_mapping = {
        'sachs': 'https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt',
        'boston_housing': 'https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/boston-housing/data/boston-housing.continuous.txt',
        'airfoil': 'https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/airfoil-self-noise/data/airfoil-self-noise.continuous.txt'
    }

    if dataset_name not in url_mapping:
        raise ValueError("Invalid dataset name")

    url = url_mapping[dataset_name]
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')  # Read content and decode to string

    # Use StringIO to turn the string content into a file-like object so numpy can read from it
    labels_array = np.genfromtxt(StringIO(content), delimiter="\t", dtype=str, max_rows=1)
    data = np.loadtxt(StringIO(content), skiprows=1)

    # Convert labels_array to a list of strings
    labels_list = labels_array.tolist()
    if isinstance(labels_list, str):  # handle the case where there's only one label
        labels_list = [labels_list]

    return data, labels_list