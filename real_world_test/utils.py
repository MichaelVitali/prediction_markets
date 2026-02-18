from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def check_array_nan(array: NDArray) -> None:
    """
    Checks if the array have only NaN values. If it has we throw an error.

    Parameters
    ----------
    array: NDArray
        an array with non-numerical or non-categorical values

    Raises
    ------
    ValueError
        If all elements of the array are NaNs
    """
    if np.isnan(array).all() and len(np.unique(array)) > 0:
        raise ValueError(
            "Array contains only NaN values."
        )
    
def check_array_inf(array: NDArray) -> None:
    """
    Checks if the array have inf.
    If a value is infinite, we throw an error.

    Parameters
    ----------
    array: NDArray
        an array with non-numerical or non-categorical values

    Raises
    ------
    ValueError
        If any elements of the array is +inf or -inf.
    """
    if np.isinf(array).any():
        raise ValueError(
            "Array contains infinite values."
        )
    
def check_arrays_length(*arrays: NDArray) -> None:
    """
    Checks if the length of all arrays given in this function are the same

    Parameters
    ----------
    *arrays: NDArray
        Arrays expected to have the same length

    Raises
    ------
    ValueError
        If the length of the arrays are different
    """
    res = [array.shape[0] for array in arrays]
    if len(np.unique(res)) > 1:
        raise ValueError(
                "There are arrays with different length"
            )

def mean_winkler_interval_score(
        y_true: NDArray,
        y_pis: NDArray,
        alpha: float
) -> float:

    # Undo any possible quantile crossing
    y_pred_low = np.minimum(y_pis[:, 0], y_pis[:, 1])
    y_pred_up = np.maximum(y_pis[:, 0], y_pis[:, 1])

    '''check_arrays_length(y_true, y_pred_low, y_pred_up)

    # Checking for NaN and inf values
    for array in (y_true, y_pred_low, y_pred_up):
        check_array_nan(array)
        check_array_inf(array)'''

    width = np.sum(y_pred_up) - np.sum(y_pred_low)
    error_above = np.sum((y_true - y_pred_up)[y_true > y_pred_up])
    error_below = np.sum((y_pred_low - y_true)[y_true < y_pred_low])
    total_error = error_above + error_below
    mwi = (width + total_error * 2 / alpha) / len(y_true)
    return mwi


def dimensionality_reduction_train(train, cols, n_components, name_comp='pca_'):

    features = [col for col in train.columns if any(sub in col for sub in cols)]

    pca_speed = PCA(n_components=n_components)
    pca_speed.fit(train[features])

    pca_features_train = pca_speed.transform(train[features])
    pca_features_train = pd.DataFrame(pca_features_train, columns=[f'{name_comp}{i}' for i in range(pca_features_train.shape[1])])
    train = pd.concat([train, pca_features_train], axis=1)
    train = train.drop(columns=features)

    return pca_speed, train

def dimensionality_reduction_test(test, pca, name_comp):

    features = pca.feature_names_in_
    
    pca_features_test = pca.transform(test[pca.feature_names_in_])
    pca_features_test = pd.DataFrame(pca_features_test, columns=[f'{name_comp}{i}' for i in range(pca_features_test.shape[1])])
    test = pd.concat([test, pca_features_test], axis=1)
    test = test.drop(columns=features)

    return test

def cycle_encoding(df, col, max_val):
    if isinstance(max_val, int):
        max_val = [max_val]
    for val in max_val:
        df[col + '_sin_' + str(val)] = np.sin(2 * np.pi * df[col]/val)
        df[col + '_cos_' + str(val)] = np.cos(2 * np.pi * df[col]/val)
    df = df.drop(columns=[col])

    return df
