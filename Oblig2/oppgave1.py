import time
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from scipy import signal
from numba import jit


#1.1
@jit
def middelverdifiltrering(filename, k):
    img = imread(filename, as_gray=True)
    filter = (1/k**2)*np.ones([k, k])
    konv = signal.convolve2d(img, filter)

    N,M = img.shape

    #nullutvidelse
    if (N != k) and (M != k) :
        if (k % 2 == 0) and ((N % 2 == 0) and (M % 2 == 0)):
            paddedfilter = zeropadding(img, k, filter)
        else:
            paddedfilter = zeropadding_odd(img, k, filter)

    #1 transformere innbildet til frekvensdomenet
    F = np.fft.fft2(img)
    #2 transformer filter til frekvensdomenet
    H = np.fft.fftshift(paddedfilter)
    H = np.fft.fft2(H)
    #3 multipliser de to transformerte matrisene
    DFT = F * H
    #4 inverser transform tilbake til bildedomenet
    IDFT = np.real(np.fft.ifft2(DFT))

    return img, konv, IDFT

@jit
def zeropadding_odd(img, k, filter):
    N,M = img.shape

    kn = (N-k)/2; km = (M-k)/2
    kn = int(np.floor(kn))
    km = int(np.floor(km))

    paddedImg = np.zeros((N, M))
    
    #kanter
    paddedImg[0:kn, M-1] = 0
    paddedImg[N-1, 0:km] = 0
    paddedImg[N-kn:N-1, M-1] = 0
    paddedImg[N-1, M-km:M-1] = 0

    paddedImg[kn:N-kn-1, km:M-km-1] = filter[:k, :k]
    
    return paddedImg

@jit
def zeropadding(img, k, filter):
    N,M = img.shape
    kn = int((N-k)/2); km = int((M-k)/2)

    paddedImg = np.zeros((N, M))

    #kanter
    paddedImg[0:kn, M-1] = 0
    paddedImg[N-1, 0:km] = 0
    paddedImg[k:N, M-1] = 0
    paddedImg[N-1, k:M-1] = 0

    #selve bildet
    paddedImg[kn:N-kn, km:M-km] = filter[:N, :M]
    
    return paddedImg

#1.2
@jit
def middelverdifiltrering2(filename, k):
    img = imread(filename, as_gray=True)
    filter = (1/k**2)*np.ones([k, k])

    starttid = time.time()
    konv = signal.convolve2d(img, filter, mode = 'same')
    tidkonv = time.time () - starttid

    N,M = img.shape

    starttid = time.time()
    #1 transformere innbildet til frekvensdomenet
    F = np.fft.fft2(img)
    #2 transformer filter til frekvensdomenet
    H = np.fft.fftshift(filter)
    H = np.fft.fft2(H, (N, M))
    #3 multipliser de to transformerte matrisene
    DFT = F * H
    #4 inverser transform tilbake til bildedomenet
    IDFT = np.real(np.fft.ifft2(DFT))
    tidDFT = time.time () - starttid
    return img, konv, IDFT, tidkonv, tidDFT
    
@jit
def kjoretid(k):
    b_plot = []
    f_plot = [] 
    for i in range(1,k):
        _,_,_, b, f = middelverdifiltrering2('cow.png', i)
        b_plot.append(b)
        f_plot.append(f)
    return b_plot, f_plot


if __name__ == '__main__':
    #Task 1.1
    
    img, konv, IDFT = middelverdifiltrering('cow.png', 15)
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Før")
    plt.subplot(1,3,2)
    plt.imshow(konv, cmap='gray')
    plt.title("Etter konvolusjon")
    plt.subplot(1,3,3)
    plt.imshow(IDFT, cmap='gray', vmin=0, vmax=255)
    plt.title("Etter fourier-transform")
    plt.show()

    #Task 1.2
    img, konv, IDFT,_,_ = middelverdifiltrering2('cow.png', 15)
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Før")
    plt.subplot(1,3,2)
    plt.imshow(konv, cmap='gray')
    plt.title("Etter konvolusjon")
    plt.subplot(1,3,3)
    plt.imshow(IDFT, cmap='gray', vmin=0, vmax=255)
    plt.title("Etter fourier-transform")
    plt.show()

    #Task 1.3
    
    b_plot, f_plot = kjoretid(30) 
    X = np.arange(1, 30, 1)
    plt.plot(X, b_plot, color='r', label='konvolusjon')
    plt.plot(X, f_plot, color='g', label='DFT')
    plt.xlabel("Filter-size")
    plt.ylabel("Time")
    plt.title("Runtime for filter size 1-30")
    plt.legend()
    plt.show()


