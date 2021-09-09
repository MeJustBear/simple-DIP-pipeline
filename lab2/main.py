import os
import cv2
import pathlib as plib

# from matplotlib import pyplot as plt

from funcs import *

height = 1024
width = 1280

sigma = [10, 10, 2, 2]
radius = 5

tasks = [0, 0, 1, 0]

oCV = True


def main_demosaicing(result_name):
    global width
    global height
    size_bauer = width * height
    frame_num = 1

    f = open('./pics/dump_white_color_10_frames.raw', 'rb').read()
    data = np.frombuffer(f, dtype=np.uint16, count=size_bauer, offset=size_bauer * frame_num * 2)

    data = np.reshape(data, [height, width])
    del f

    rgbTriple = demosaicing(data, height, width)
    plt.imsave(result_name, rgbTriple.astype(np.float32) / 4095.0)
    return rgbTriple


def main_white_balance(rgbTriple, filename="awb"):
    result_patch = white_patch(rgbTriple)
    result_gray = grayworld(rgbTriple)
    result_comb = awb_comb(rgbTriple)
    plt.imsave(filename + "_white_patch.png", result_patch.astype(np.float32) / 4095.0)
    plt.imsave(filename + "_grayworld.png", result_gray.astype(np.float32) / 4095.0)
    plt.imsave(filename + "_combine.png", result_comb.astype(np.float32) / 4095.0)

    np.save(filename + "_white_patch", result_patch)
    np.save(filename + "_grayworld", result_gray)
    np.save(filename + "_combine", result_comb)


def main_third_task(rgbTriple, gamma, coefA=1.0, filename=""):
    global isYCbCr, oCV
    ycbcr = rgb2ycbcr(rgbTriple.astype(np.float32) / 4095)
    y_comp = np.copy(ycbcr)
    y_comp[:, :, 1] = y_comp[:, :, 0]
    y_comp[:, :, 2] = y_comp[:, :, 0]
    plt.imsave(filename + "_y.png", y_comp)

    result_gamma = gamma_correction(ycbcr, gamma, coefA)
    print("Gamma correction done")

    result_hist = hist_equalise(np.round(result_gamma * 4095).astype(np.uint16), 12)
    result_hist = result_hist.astype(np.float32) / 4095
    # result_hist = ycbcr2rgb(result_hist)

    y_comp = np.zeros([len(result_gamma), len(result_gamma[0]), 3], dtype=np.float32)

    y_comp[:, :, 0] = result_gamma[:, :]
    y_comp[:, :, 1] = y_comp[:, :, 0]
    y_comp[:, :, 2] = y_comp[:, :, 0]
    plt.imsave(filename + "_gamma_correction_y.png", y_comp)
    y_comp[:, :, 1] = ycbcr[:, :, 1]
    y_comp[:, :, 2] = ycbcr[:, :, 2]
    plt.imsave(filename + "_gamma_correction.png", ycbcr2rgb(y_comp))
    if (oCV):
        img = cv2.imread(filename + '_gamma_correction.png', 0)
        res = cv2.equalizeHist(img)
        cv2.imwrite(filename + "_cv_hist.png", res)

    y_comp[:, :, 0] = result_hist[:, :]

    y_comp[:, :, 1] = y_comp[:, :, 0]
    y_comp[:, :, 2] = y_comp[:, :, 0]
    plt.imsave(filename + "_histogram_equalisation_y.png", y_comp)
    y_comp[:, :, 1] = ycbcr[:, :, 1]
    y_comp[:, :, 2] = ycbcr[:, :, 2]
    plt.imsave(filename + "_histogram_equalisation.png", ycbcr2rgb(y_comp))

    np.save(filename + "_gamma_correction", result_gamma)
    np.save(filename + "_histogram_equalisation", ycbcr2rgb(y_comp))


def main_four_task(rgbTriple, sigma, radius, filename=""):
    res = cv2.GaussianBlur(rgbTriple, (radius, radius), sigmaX=sigma[0], sigmaY=sigma[1])
    plt.imsave(filename + "cv_gaussian.png", res)

    result_gaussian = gaussian_filter(rgbTriple, sigmaX=sigma[0], sigmaY=sigma[1], radius=radius)
    plt.imsave(filename + "gaussian.png", result_gaussian)

    res = cv2.bilateralFilter(rgbTriple, 0, sigma[2], sigma[3])
    plt.imsave(filename + "cv_bilateral.png", res)

    result_bilateral = bilateral_filter(result_gaussian[560:580, 890:910, :], sigma[2], sigma[3])
    plt.imsave(filename + "_bilateral.png", result_bilateral)


def main():

    absPath = plib.Path().absolute()


    if tasks[0]:
        print("Start demosaicing")
        result_name = "result_demosaicing.png"
        curPath = plib.Path.joinpath(absPath, "1Task")
        os.makedirs(curPath, exist_ok=True)
        rgb = main_demosaicing(str(plib.Path.joinpath(curPath, result_name)))
        np.save(str(curPath.parent.joinpath(curPath, "rgb_data")), rgb)
        print("Demosaicing done")

    if tasks[1]:
        with open("./1Task/rgb_data.npy", "rb") as f:
            rgb = np.load(f)
        print("start awb")
        curPath = plib.Path.joinpath(absPath, "2Task")
        os.makedirs(curPath, exist_ok=True)
        curPath = plib.Path.joinpath(curPath, "awb")
        main_white_balance(rgb, filename=str(curPath))
        print("Awb done")

    if tasks[2]:
        with open("./2Task/awb_combine.npy", "rb") as f:
            awb = np.load(f)
        print("Start gamma and hist eqv")
        curPath = plib.Path.joinpath(absPath, "3Task")
        os.makedirs(curPath, exist_ok=True)
        main_third_task(awb, 1 / 2.2, coefA=1.0, filename=str(curPath) + '/')
        print("Gamma correction and histogram equalisation done")

    if tasks[3]:
        with open("./3Task/_histogram_equalisation.npy", "rb") as f:
            gc = np.load(f)
        print("Start filtering")
        curPath = plib.Path.joinpath(absPath, "4Task")
        os.makedirs(curPath, exist_ok=True)
        main_four_task(gc, sigma, radius, filename=str(curPath) + '/')
        print("filtering done")
    plt.show()


main()
