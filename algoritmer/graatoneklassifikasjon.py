from imageio import imread
import numpy as np
from histogram import *
import matplotlib.pyplot as plt


def graatoneklass():
    # Load a "clean" image
    im_clean = imread('bilder/textImage_clean.png', as_gray=True)
    (N, M) = im_clean.shape
    #lagHist('textImage_clean.png')

    # Add some noise
    noiseStd = 30 # How much white noise to add
    im_noisy = im_clean + noiseStd * np.random.normal(0, 1, (N,M))
    #lagHistforBildet(im_noisy)

    # Add varying light-intesity model
    lightFactor = 500 # Increasing this increases effect of our varying-light model
    lightMask = np.array([[(x-M/2)/M for x in range(1, M+1)] for y in range(N)])
    im_light = im_clean + lightFactor * lightMask
    #lagHistforBildet(im_light)

    # Separate background and foreground pixels using our "clean" image
    backgroundPixels = [im_noisy[x][y] for x in range(N) for y in range(M) if im_clean[x][y] < 150]
    foregroundPixels = [im_noisy[x][y] for x in range(N) for y in range(M) if im_clean[x][y] > 150]
    #plot2Hist(backgroundPixels, foregroundPixels)
    #antall = klassifiseringsfeil(backgroundPixels, foregroundPixels, 175)
    #print("antall: ", antall)
    normaliser2Hist(backgroundPixels, foregroundPixels)

def klassifiseringsfeil(img_b, img_f, t):
    out1, bins, _ = plt.hist(img_b)
    out2, bins, _= plt.hist(img_f)
    antall = sum(out1[int(out2.min()), t]*np.diff(bins[int(out2.min()), t+1]))
    # for x in range(int(out2.min()), t):
    #     antall += out1[x]
    # for x in range(t, out1.max()):
    #     antall += out2[x]
    return antall


if __name__ == '__main__':
    graatoneklass()