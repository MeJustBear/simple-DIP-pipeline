import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

nfig = 1


def _calc(x, y, sigmaX, sigmaY):
    return 1/(2 * np.pi*sigmaX * sigmaY)*np.exp(-(x**2 + y**2)/(2*sigmaX * sigmaY))


def _calc_gauss(x, sigma):
    return 1/(2 * np.pi*sigma**2)*np.exp(-(x**2)/(2*sigma**2))


def _calc_euclid(x, y):
    return np.sqrt(((x[0] - x[1])**2 + (y[0] - y[1])**2))


def demosaicing(data, height, width):
    """ 3d array of demosaicing result 0-Red 1- Green 2- Blue """
    result = np.zeros([height, width, 3], dtype=np.uint16)

    for i in range(0, height, 2):
        coefficients_blue = [1, 1, 1]
        coefficients_red = [1, 1, 1]
        coefficients_green = [1, 1, 1, 1]
        for j in range(0, width, 2):

            if j + 2 == width:
                coefficients_red[0] = 0
                coefficients_red[1] = 0
                coefficients_green[0] = 0
            if i + 2 == height:
                coefficients_red[2] = 0
                coefficients_red[0] = 0
                coefficients_green[1] = 0

            if i == 0:
                coefficients_blue[0] = 0
                coefficients_blue[2] = 0
                coefficients_green[2] = 0
            if j == 0:
                coefficients_blue[1] = 0
                coefficients_blue[0] = 0
                coefficients_green[3] = 0

            # Red block
            result[i][j][0] = data[i][j]
            tmp = (data[i][j] + coefficients_red[0] * data[coefficients_red[0] * (i + 2)][coefficients_red[0] * (j + 2)]
                   + coefficients_red[2] * data[coefficients_red[2] * (i + 2)][j] + coefficients_red[1] * data[i][
                       coefficients_red[1] * (j + 2)]) / (1 + sum(coefficients_red))
            result[i + 1][j + 1][0] = tmp
            tmp = (data[i][j] + tmp + coefficients_red[1] * data[i][coefficients_red[1] * (j + 2)]) / (
                    2 + coefficients_red[1])
            result[i][j + 1][0] = tmp
            tmp = (data[i][j] + coefficients_red[2] * data[coefficients_red[2] * (i + 2)][j]) / (
                    1 + coefficients_red[2])
            result[i + 1][j][0] = tmp

            # Blue block
            result[i + 1][j + 1][2] = data[i + 1][j + 1]
            tmp1 = (data[i + 1][j + 1] +
                    coefficients_blue[0] * data[coefficients_blue[0] * (i - 1)][coefficients_blue[0] * (j - 1)] +
                    coefficients_blue[1] * data[coefficients_blue[1] * (i + 1)][coefficients_blue[1] * (j - 1)] +
                    coefficients_blue[2] * data[coefficients_blue[2] * (i - 1)][coefficients_blue[2] * (j + 1)]) / \
                   (sum(coefficients_blue) + 1)
            result[i][j][2] = tmp1
            tmp = (tmp1 + data[i + 1][j + 1] +
                   coefficients_blue[1] * data[coefficients_blue[1] * (i + 1)][coefficients_blue[1] * (j - 1)]) / \
                  (2 + coefficients_blue[1])
            result[i + 1][j][2] = tmp
            tmp = (tmp1 + data[i + 1][j + 1] + coefficients_blue[2] *
                   data[coefficients_blue[2] * (i - 1)][coefficients_blue[2] * (j + 1)]) / (2 + coefficients_blue[2])
            result[i][j + 1][2] = tmp

            # Green block
            result[i][j + 1][1] = data[i][j + 1]
            result[i + 1][j][1] = data[i + 1][j]
            tmp = (data[i][j + 1] + data[i + 1][j] +
                   coefficients_green[0] * data[coefficients_green[0] * (i + 1)][coefficients_green[0] * (j + 2)] +
                   coefficients_green[1] * data[coefficients_green[1] * (i + 2)][coefficients_green[1] * (j + 1)]) / \
                  (2 + coefficients_green[0] + coefficients_green[1])
            result[i + 1][j + 1][1] = tmp
            tmp = (data[i + 1][j] + data[i][j + 1] +
                   coefficients_green[2] * int(data[coefficients_green[2] * i][coefficients_green[2] * (j - 1)]) +
                   coefficients_green[3] * int(data[coefficients_green[3] * (i - 1)][coefficients_green[3] * j])) / \
                  (2 + coefficients_green[2] + coefficients_green[3])
            result[i][j][1] = tmp

    return result


def white_patch(rgbTriple):
    """ white patch algorythm for awb rgbTriple M x N x 3 np array """
    avgTuple = np.average(np.average(rgbTriple, axis=0), axis=0)

    alpha = avgTuple[1] / avgTuple[0]
    beta = avgTuple[1] / avgTuple[2]

    result = np.round(np.prod([rgbTriple, [alpha, 1, beta]]))

    return result.astype(np.uint16)


def grayworld(rgbTriple):
    """ grayworld algorythm for awb rgbTriple M x N x 3 np array """
    maxTuple = np.max(np.max(rgbTriple, axis=0), axis=0)

    alpha = maxTuple[1] / maxTuple[0]
    beta = maxTuple[1] / maxTuple[2]

    result = np.round(np.prod([rgbTriple, [alpha, 1, beta]]))

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

    result = np.prod([rgbTriple ** 2, [cR[0], 0, cB[0]]])
    res = np.prod([rgbTriple, [cR[1], 1, cB[1]]])
    result = res + result

    return result.astype(np.uint16)


def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 0.5
    np.clip(ycbcr, 0, 1, out=ycbcr)
    return np.float32(ycbcr)


def ycbcr2rgb(image):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = image.astype(np.float)
    rgb[:, :, [1, 2]] -= 0.5
    rgb = rgb.dot(xform.T)
    np.clip(rgb, 0, 1, out=rgb)
    return np.float32(rgb)


def gamma_correction(colorDif, gamma, coefA=1.0):
    colorDif = colorDif[:, :, 0] ** gamma
    result = np.prod([colorDif, coefA])
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _histogram(array, title, maxBin):
    global nfig
    f = plt.figure(nfig)
    nfig += 1

    plt.title(title)
    histogram = np.histogram(array, bins=np.arange(0, maxBin), density=True)
    plt.bar(histogram[1][:-1], histogram[0][:], color="r")
    checksum = histogram[0].sum()
    if checksum > 1:
        raise Exception("Incorrect checksum")
    print("Sum hist = ", checksum)
    plt.grid("on")
    #plt.legend()
    return histogram[0][:]


def hist_equalise(y, l):
    """
    histogram equalisation algorithm
    rgbTriple: M x N x 3 np array
    l: bits for value
    mn: area of image
    """
    l = pow(2, l)
    res = np.ndarray.copy(y)

    print("Before equalise")
    hist = _histogram(y, "histogram before", (l + 2))
    print("Start equalise")

    np.cumsum(hist, out=hist)
    hist = np.uint16(np.round(np.prod([hist, (l - 1)])))

    for i in range(len(res)):
        for j in range(len(res[0])):
            try:
                res[i][j] = hist[res[i][j]]
            except Exception as e:
                print("err")
    _histogram(res, "histogram after", (l + 2))

    return res


def gaussian_filter(rgbTriple, sigmaX, sigmaY, radius):
    if not radius % 2:
        raise Exception("incorrect raduis value")

    kernel = np.zeros([radius, radius], dtype=np.float32)
    ycbcr = rgb2ycbcr(rgbTriple)

    for i in range(radius):
        for j in range(radius):
            kernel[i][j] = _calc(i - (radius - 1) / 2, j - (radius - 1) / 2, sigmaX, sigmaY)

    kernel = kernel * (1 / np.sum(kernel))

    ycbcr[:, :, 0] = signal.convolve2d(ycbcr[:, :, 0], kernel, mode="same", boundary="symm", fillvalue=0)

    return ycbcr2rgb(ycbcr)


def bilateral_filter(rgbTriple, sigmaS, sigmaR):
    ycbcr = rgb2ycbcr(rgbTriple)
    copyLuma = np.copy(ycbcr[:, :, 0])

    for i in range(len(copyLuma)):
        for j in range(len(copyLuma[0])):
            currentSum = 0
            Wp = 0
            for k in range(len(copyLuma)):
                for l in range(len(copyLuma[0])):
                    currGaus = _calc_gauss(_calc_euclid([i, k], [j, l]), sigmaS) * _calc_gauss(ycbcr[i, j, 0] - ycbcr[k, l, 0], sigmaR)
                    Wp += currGaus
                    currentSum += currGaus * ycbcr[k, l, 0]
            copyLuma[i, j] = (1 / Wp) * currentSum
    ycbcr[:, :, 0] = copyLuma

    return ycbcr2rgb(ycbcr)
