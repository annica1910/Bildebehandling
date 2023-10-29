import math
import matplotlib.pyplot as plt
import numpy as np
from histutjevning import *

#Histogramutjevning for et N x M bilde 
#Transformerer trinnvis
def histogramtilpasning(filename):
    T = histogramutjevning(filename)

    plt.hist(T)
    plt.title("Equalized histogram of " + filename)
    plt.show()

histogramutjevning('car.png')
