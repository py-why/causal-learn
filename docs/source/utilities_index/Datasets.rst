.. _datasets:

Datasets
==============================================



Usage
----------------------------
.. code-block:: python

    from causallearn.utils.Dataset import load_dataset

    data, labels = load_dataset(dataset_name)

Parameters
-------------------
**dataset_name**: str, the name of a dataset in ['sachs', 'boston_housing', 'airfoil']


Returns
-------------------

**data**: np.array, data with a shape of (number of samples, number of variables).

**labels**: list, labels of variables in the data.

