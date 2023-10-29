import math
import matplotlib.pyplot as plt
import numpy as np

#Histogramutjevning for et N x M bilde 
#Transformerer trinnvis
def histogramutjevning(filename):
    img = plt.imread(filename)
    plt.imshow(img, cmap = "gray")
    plt.figure()
    h, bins = plt.hist(img)
    N,M = img.shape
    plt.show()
    plt.title("Histogram of " + filename)

    p, c, T = h
    c[0] = p[0]
    for i in len(h):
        h[i] = h[i] + 1
        p[i] = h[i]/(N*M)
        c[i] = c[i-1]+p[i]
        T[i] = math.ceil((bins-1)*c[i])
    
    plt.figure()
    plt.hist(p)
    plt.title("Normalised histogram of " + filename)
    plt.figure()
    plt.hist(c)
    plt.title("Normalised cumulative histogram of " + filename)
    plt.figure()
    plt.hist(T)
    plt.title("Equalized histogram of " + filename)

    return T

histogramutjevning('bilder/car.png')
