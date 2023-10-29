import matplotlib.pyplot as plt
import numpy as np

#Et histogram for et bilde N x M der I trinnvis øker fra Imin til Imax
#N = kol, M = rad, k = bits, 
def lagHist(filename):
    img = plt.imread(filename)
    plt.figure()
    plt.hist(img)
    plt.title("Histogram of" + filename)
    plt.show()

def lagHistforBildet(img):
    plt.figure()
    plt.hist(img)
    plt.show()

def plot2Hist(a, b):
    #plt.hist([a, b], label =['h1', 'h2'])
    plt.hist(a, alpha = 0.5, label='h1')
    plt.hist(b, alpha = 0.5, label='h2')
    plt.legend(loc='upper left')
    plt.show()

def normaliser2Hist(a, b):
    plt.hist(a, density=True, alpha = 0.5, label='p1')
    plt.hist(b, density= True, alpha = 0.5, label='p2')
    plt.legend(loc='upper left')
    plt.show()


