"""This module contains three sorting function: lexicographic sort of atomic coordinates,
sort that uses projections on a vector pointing from one electrode to another as the sorting keys
and sort that uses a potential function over atomic coordinates as the sorting keys.
A user can define his own sorting procedure - the user-defined sorting function should contain
`**kwargs` in the list of arguments and it can uses in its body one of the arguments with following name convention:
`coords` is the list of atomic coordinates,
`left_lead` is the list of the indices of the atoms contacting the left lead,
`right_lead` is the list of the indices of the atoms contacting the right lead, and
`mat` is the adjacency matrix of the tight-binding model.
All functions return the list of sorted atomic indices.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lgmres


def sort_lexico(coords=None, **kwargs):
    """Lexicographic sort

    Parameters
    ----------
    coords : array
        list of atomic coordinates (Default value = None)
    **kwargs :
        

    Returns
    -------

    
    """
    return np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))


def sort_projection(coords=None, left_lead=None, right_lead=None, **kwargs):
    """Sorting procedure that uses projections on a vector pointing from one electrode to another as the sorting keys.

    Parameters
    ----------
    coords : array
        list of atomic coordinates (Default value = None)
    left_lead : array
        list of the atom indices contacting the left lead (Default value = None)
    right_lead : array
        list of the atom indices contacting the right lead (Default value = None)
    **kwargs :
        

    Returns
    -------

    
    """
    vec = np.mean(coords[left_lead], axis=0) - np.mean(coords[right_lead], axis=0)
    keys = np.dot(coords, vec) / np.linalg.norm(vec)

    return np.argsort(keys, kind='mergesort')


def sort_capacitance(coords, mat, left_lead, right_lead, **kwargs):
    """Sorting procedure that uses a potential function defined over atomic coordinates as the sorting keys.

    Parameters
    ----------
    coords : array
        list of atomic coordinates
    mat : 2D array
        adjacency matrix of the tight-binding model
    left_lead : array
        list of the atom indices contacting the left lead
    right_lead : array
        list of the atom indices contacting the right lead
    **kwargs :
        

    Returns
    -------

    
    """

    charge = np.zeros(coords.shape[0], dtype=complex)
    charge[left_lead] = 1e3
    charge[right_lead] = -1e3

    x = coords[:, 1].T
    y = coords[:, 0].T

    mat = (mat != 0.0).astype(float)
    mat = 10 * (mat - np.diag(np.diag(mat)))
    mat = mat - np.diag(np.sum(mat, axis=1)) + 0.001 * np.identity(mat.shape[0])

    col, info = lgmres(mat, charge.T, x0=1.0 / np.diag(mat), tol=1e-5, maxiter=15)
    col = col / np.max(col)

    indices = np.argsort(col, kind='heapsort')

    mat = mat[indices, :]
    mat = mat[:, indices]

    plt.scatter(x, y, c=col, cmap=plt.cm.get_cmap('seismic'), s=50, marker="o", edgecolors="k")
    plt.colorbar()
    plt.axis('off')
    plt.show()

    return indices
