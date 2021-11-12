.. _lingam:

LiNGAM-based Methods
============================

Estimation of Linear, Non-Gaussian Acyclic Model from observed data. It assumes non-Gaussianity of the noise terms in the causal model.

causal-learn has the official implementations for a set of LiNGAM-based methods (e.g., ICA-based LiNGAM [1]_, DirectLiNGAM [2]_, VAR-LiNGAM [3]_, RCD [4]_, and CAM-UV [5]_).
And we are actively updating the list.

.. [1] Shimizu, S., Hoyer, P. O., Hyvärinen, A., Kerminen, A., & Jordan, M. (2006). A linear non-Gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7(10).
.. [2] Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T., ... & Bollen, K. (2011). DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model. The Journal of Machine Learning Research, 12, 1225-1248.
.. [3] Hyvärinen, A., Zhang, K., Shimizu, S., & Hoyer, P. O. (2010). Estimation of a structural vector autoregression model using non-gaussianity. Journal of Machine Learning Research, 11(5).
.. [4] Maeda, T. N., & Shimizu, S. (2020, June). RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent confounders. In International Conference on Artificial Intelligence and Statistics (pp. 735-745). PMLR.
.. [5] Maeda, T. N., & Shimizu, S. (2021). Causal Additive Models with Unobserved Variables. UAI.

ICA-based LiNGAM
--------------------------------------

.. code-block:: python

    from causallearn.search.FCMBased import lingam
    model = lingam.ICALiNGAM(random_state, max_iter)
    model.fit(X)

    print(model.causal_order_)
    print(model.adjacency_matrix_)


Parameters
""""""""""""""""""""""""""""""""""""

**random_state**: int, optional (default=None). The seed used by the random number generator.

**max_iter**: int, optional (default=1000). The maximum number of iterations of FastICA.

**X**: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
and n_features is the number of features.

Returns
""""""""""""""""""""""""""""""""""""

**model.causal_order_**: array-like, shape (n_features).
The causal order of fitted model, where n_features is the number of features.

**model.adjacency_matrix_**: array-like, shape (n_features, n_features).
The adjacency matrix B of fitted model, where n_features is the number of features.



DirectLiNGAM
--------------------------------------

.. code-block:: python

    from causallearn.search.FCMBased import lingam
    model = lingam.DirectLiNGAM(random_state, prior_knowledge, apply_prior_knowledge_softly, measure)
    model.fit(X)

    print(model.causal_order_)
    print(model.adjacency_matrix_)

Parameters
""""""""""""""""""""""""""""""""""""

**random_state**: int, optional (default=None). The seed used by the random number generator.

**prior_knowledge**: array-like, shape (n_features, n_features), optional (default=None).
Prior knowledge used for causal discovery, where ``n_features`` is the number of features.
The elements of prior knowledge matrix are defined as follows:
    - 0: :math:`x_i` does not have a directed path to :math:`x_j`
    - 1: :math:`x_i` has a directed path to :math:`x_j`
    - -1: No prior knowledge is available to know if either of the two cases above (0 or 1) is true.

**apply_prior_knowledge_softly**: boolean, optional (default=False). If True, apply prior knowledge softly.

**measure**: {'pwling', 'kernel'}, optional (default='pwling'). Measure to evaluate independence: 'pwling' or 'kernel'.

**X**: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
and n_features is the number of features.

Returns
""""""""""""""""""""""""""""""""""""

**model.causal_order_**: array-like, shape (n_features).
The causal order of fitted model, where n_features is the number of features.

**model.adjacency_matrix_**: array-like, shape (n_features, n_features).
The adjacency matrix B of fitted model, where n_features is the number of features.



VAR-LiNGAM
--------------------------------------

.. code-block:: python

    from causallearn.search.FCMBased import lingam
    model = lingam.VARLiNGAM(lags, criterion, prune, ar_coefs, lingam_model, random_state)
    model.fit(X)

    print(model.causal_order_)
    print(model.adjacency_matrices_[0])
    print(model.adjacency_matrices_[1])
    print(model.residuals_)

Parameters
""""""""""""""""""""""""""""""""""""

**lags**: int, optional (default=1). Number of lags.

**criterion**: {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic'). Criterion to decide the best lags within 'lags'. Searching the best lags is disabled if 'criterion' is None.

**prune**: boolean, optional (default=False). Whether to prune the adjacency matrix or not.

**ar_coefs**: array-like, optional (default=None). Coefficients of AR model. Estimating AR model is skipped if specified 'ar_coefs'. Shape must be ('lags', n_features, n_features).

**lingam_model**: lingam object inherits 'lingam._BaseLiNGAM', optional (default=None). LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.

**random_state**: int, optional (default=None). 'random_state' is the seed used by the random number generator.

**X**: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
and n_features is the number of features.

Returns
""""""""""""""""""""""""""""""""""""

**model.causal_order_**: array-like, shape (n_features).
The causal order of fitted model, where n_features is the number of features.

**model.adjacency_matrices_**: array-like, shape (lags, n_features, n_features).
The adjacency matrix of fitted model, where n_features is the number of features.

**model.residuals_**: array-like, shape (n_samples).
Residuals of regression, where n_samples is the number of samples.


RCD
--------------------------------------

.. code-block:: python

    from causallearn.search.FCMBased import lingam
    model = lingam.RCD(max_explanatory_num, cor_alpha, ind_alpha, shapiro_alpha, MLHSICR, bw_method)
    model.fit(X)

    print(model.adjacency_matrix_)
    print(model.ancestors_list_)

Parameters
""""""""""""""""""""""""""""""""""""

**max_explanatory_num**: int, optional (default=2). Maximum number of explanatory variables.

**cor_alpha**: float, optional (default=0.01). Alpha level for pearson correlation.

**ind_alpha**: float, optional (default=0.01). Alpha level for HSIC.

**shapiro_alpha**: float, optional (default=0.01). Alpha level for Shapiro-Wilk test.

**MLHSICR**: bool, optional (default=False). If True, use MLHSICR for multiple regression, if False, use OLS for multiple regression.

**bw_method**: str, optional (default='mdbs'). The method used to calculate the bandwidth of the HSIC.
    - 'mdbs': Median distance between samples.
    - 'scott': Scott's Rule of Thumb.
    - 'silverman': Silverman's Rule of Thumb.

**X**: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
and n_features is the number of features.

Returns
""""""""""""""""""""""""""""""""""""

**model.adjacency_matrix_**: array-like, shape (n_features, n_features).
The adjacency matrix B of fitted model, where n_features is the number of features.

**model.ancestors_list_**: array-like, shape (n_features).
The list of causal ancestors sets, where n_features is the number of features.


CAM-UV
--------------------------------------

.. code-block:: python

    from causallearn.search.FCMBased.lingam import CAMUV
    P, U = CAMUV.execute(data, alpha, num_explanatory_vals)

    for i, result in enumerate(P):
        if not len(result) == 0:
            print("child: " + str(i) + ",  parents: " + str(result))

    for result in U:
        print(result)

Parameters
""""""""""""""""""""""""""""""""""""

**data**: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
and n_features is the number of features.

**alpha**: the alpha level for independence testing.

**num_explanatory_vals**: the maximum number of variables to infer causal relationships. This is equivalent to d in the paper.

Returns
""""""""""""""""""""""""""""""""""""

**P**: P[i] contains the indices of the parents of Xi.

**U**: The indices of variable pairs having UCPs or UBPs.


