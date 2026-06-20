"""Automatic white balance algorithms.

Migrated verbatim from the original ``funcs.py`` lab code; logic unchanged.
"""

import numpy as np


def white_patch(rgbTriple):
    """ white patch algorythm for awb rgbTriple M x N x 3 np array """
    avgTuple = np.average(np.average(rgbTriple, axis=0), axis=0)

    alpha = avgTuple[1] / avgTuple[0]
    beta = avgTuple[1] / avgTuple[2]

    result = np.round(rgbTriple * [alpha, 1, beta])

    return result.astype(np.uint16)


def grayworld(rgbTriple):
    """ grayworld algorythm for awb rgbTriple M x N x 3 np array """
    maxTuple = np.max(np.max(rgbTriple, axis=0), axis=0)

    alpha = maxTuple[1] / maxTuple[0]
    beta = maxTuple[1] / maxTuple[2]

    result = np.round(rgbTriple * [alpha, 1, beta])

    return result.astype(np.uint16)


def awb_comb(rgbTriple):
    """ combine algorythm for awb rgbTriple- M x N x 3 np array """
    sumSquads = np.sum(np.sum(rgbTriple ** 2, axis=0), axis=0)
    sums = np.sum(np.sum(rgbTriple, axis=0), axis=0)
    maxTuple = np.max(np.max(rgbTriple, axis=0), axis=0)

    cR = np.zeros([1, 2])
    cB = np.zeros([1, 2])

    N = np.zeros(2)
    N[0] = sums[1]
    N[1] = maxTuple[1]

    for i in range(2):
        M = np.zeros([2, 2])
        M[0][0] = sumSquads[i * 2]
        M[0][1] = sums[i * 2]
        M[1][0] = maxTuple[i * 2] ** 2
        M[1][1] = maxTuple[i * 2]
        if i == 0:
            cR = np.linalg.solve(M, N)
        else:
            cB = np.linalg.solve(M, N)

    result = rgbTriple ** 2 * [cR[0], 0, cB[0]]
    res = rgbTriple * [cR[1], 1, cB[1]]
    result = res + result

    return result.astype(np.uint16)
