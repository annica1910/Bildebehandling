import math
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import sympy as sp
from numba import jit

#detektere kantene til cellekjernene i bildet
#strukturelement: rektangulært naboskap med odde lengder
@jit
def konvolusjon(img, filter):
    plt.imshow(img, cmap = 'gray')
    N,M = img.shape
    n,m = filter.shape
    km = int(np.floor(m/2))
    kn = int(np.floor(n/2))
    paddedImg, N2, M2 = padding(img, km, kn)

    newFilter = np.zeros((n,m))
    newFilter = filter[::-1,::-1]

    output = np.zeros((N,M))
    row = 0
    for i in range(N2+1-n):
        for j in range(M2+1-m):
            sum = 0
            while(row < n):
                r = newFilter[row][:m]         #hver rad i filteret
                s = paddedImg[i+row][j:j+m] #neste rad
                sum += np.sum(r*s)
                row += 1
            output[i, j] = sum
            row = 0
    return output
       
def padding(img, km, kn):
    N,M = img.shape

    N2 = N + kn*2
    M2 = M + km*2
    paddedImg = np.zeros((N2, M2))

    #hjorner
    v1 = img[0,0]; h1 = img[0, M-1]
    v2 = img[N-1, 0]; h2 = img[N-1, M-1]

    #hjorner
    for i in range(kn):
        for j in range(0, km):
            paddedImg[i:kn, j:km] = v1
            paddedImg[N2-kn:N2-i, j:km] = v2
            paddedImg[i:kn, M2-km:M2-j] = h1
            paddedImg[N2-kn:N2-i, M2-km+j:M2-j] = h2

    #selve bildet
    for i in range(N):
        paddedImg[i+kn, km:M2-km] = img[i, :M]
    
    #kanter
    paddedImg[:kn, km:M2-km] = img[0, :M]
    paddedImg[N2-kn:N2, km:M2-km] = img[N-1, :M]

    for i in range(km):
        paddedImg[kn:N2-kn, i] = img[:N, 0]
        paddedImg[kn:N2-kn, M2-km+i] = img[0:N, M-1]

    return paddedImg, N2, M2

@jit
def cannys(img, sigma, T_h, T_l):
    G = gaussfilter(sigma)
    img_s = konvolusjon(img, G)
    M, a = gradientmagnitude(img_s)
    tynning = nonmax_suppression(M, a)  
    hyst = hysteresis(tynning, T_h, T_l)

    plt.subplot(2,3,1)
    plt.imshow(img, cmap = 'gray')
    plt.title("Original")
    plt.subplot(2,3,2)
    plt.imshow(img_s, cmap = 'gray')
    plt.title("Gauss-filter")
    plt.subplot(2,3,3)
    plt.imshow(M, cmap = 'gray')
    plt.title("Gradientmagnitude")
    plt.subplot(2,3,4)
    plt.imshow(a, cmap = 'gray')
    plt.title("Vinkler")
    plt.subplot(2,3,5)
    plt.imshow(tynning, cmap = 'gray')
    plt.title("Tynning")
    plt.show()

    plt.imshow(hyst, cmap = 'gray')
    plt.title("Resultat med sigma=5, T_h=70, T_l=40")
    plt.show()

@jit
def nonmax_suppression(img, a):
    M, N = img.shape
    output = np.zeros((M,N), dtype=np.int32)
    a = (a * 180)/ np.pi
    a[a < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
                pixel_bef = 0
                pixel_aft = 0
                angle = a[i][j]
                
               #angle 0
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    pixel_bef = img[i, j+1]
                    pixel_aft = img[i, j-1]
                #angle 45
                elif (22.5 <= angle < 67.5):
                    pixel_bef = img[i+1, j-1]
                    pixel_aft = img[i-1, j+1]
                elif (67.5 <= angle < 112.5):
                    pixel_bef = img[i+1, j]
                    pixel_aft = img[i-1, j]
                #angle 135
                elif (112.5 <= angle < 157.5):
                    pixel_bef = img[i-1, j-1]
                    pixel_aft = img[i+1, j+1]

                if (img[i,j] >= pixel_bef) and (img[i,j] >= pixel_aft):
                    output[i,j] = img[i,j]
                else:
                    output[i,j] = 0    
    return output

@jit
def hysteresis(img_t, T_h, T_l):
    N,M = img.shape  
    marked = np.zeros((N,M))
    marked2 = np.zeros((N,M))
    
    while(1):
        for i in range(N-1):
            for j in range(M-1):

                if img_t[i,j] >= T_h:
                    marked[i,j] = 255

                elif img_t[i,j] >= T_l and img_t[i,j] < T_h:

                    if ((marked[i+1, j-1] == 255) or (marked[i+1, j] == 255) or (marked[i+1, j+1] == 255)
                        or (marked[i, j-1] == 255) or (marked[i, j+1] == 255) or (marked[i-1, j-1] == 255) 
                        or (marked[i-1, j] == 255) or (marked[i-1, j+1] == 255)):
                            marked[i,j] = 255
                        
                else:
                    marked[i][j] = 0

        if np.array_equal(marked, marked2):
            break
        marked2=marked

    return marked

def gradientmagnitude(f_s):
    sobelX = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], 
                       [0, 0, 0], 
                       [-1, -2, -1]])
    
    gx = konvolusjon(f_s, sobelX)
    gy = konvolusjon(f_s, sobelY)

    M = np.sqrt(gx**2+gy**2)
    a = np.arctan2(gy,gx)
    return M, a

def gaussfilter(sigma):
    n = np.ceil(1 + 8*sigma)/2
    x,y = np.mgrid[-n:n+1, -n:n+1]
    A = 1/(2.0*np.pi*sigma**2)
    gauss = A*np.exp(-(x**2+y**2)/(2*sigma**2))
    return gauss

if __name__ == '__main__':       
    filename = 'cellekjerner.png'
    img = imread(filename, as_gray=True)
    cannys(img, 1.87, 105, 75)