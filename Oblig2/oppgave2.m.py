import numpy as np
from numpy import cos, pi, log
import matplotlib.pyplot as plt
from imageio import imread
from numba import jit

#Parametre:
#1. Et filnavn som spesifiserer plasseringen til bildefilen som skal benyttes
#2. Ett tall q som indirekte vil bestemme kompresjonsraten = b/c
#komprimerer bildet med kvantifisering
@jit
def bildekompresjon(img, Q, q):
    N,M = img.shape
    img -= 128
    img_out = np.zeros((N,M))

    #løper gjennom bildet og deler inn i 8x8 blokker
    for a in range(0, N, 8):
        for b in range(0, M, 8):
                block = img[a : a + 8, b : b + 8]
                for u in range(8):  #går gjennom selve blokken
                    for v in range(8):
                        k = 1/4*c(u)*c(v)
                        sum = 0
                        for x in range(8): #looper igjen pikslene i blokken med hovedpiksel som (u,v)
                            for y in range(8):
                                sum += k*block[x,y]*cos(((2*x+1)*u*pi)/16)*cos(((2*y+1)*v*pi)/16) # 
                        img_out[a+u, b+v] = np.round(sum / (Q[u,v]*q)) #kvantifiserer summen etter DFT av piksel (a+u, b+v)
    return img_out

#regner ut c for u og v
@jit
def c(a):
    if a == 0: return 1/np.sqrt(2)
    else: return 1

#motsatt av bildekompresjon
@jit
def rekonstruksjon(img, Q, q):
    N,M = img.shape
    img_out = np.zeros((N,M))

    for a in range(0, N, 8):
        for b in range(0, M, 8):
                block = img[a : a + 8, b : b + 8]
                block *= (q*Q)  #rekvantifisering
                for x in range(8):
                    for y in range(8):
                        sum = 0
                        for u in range(8):
                            for v in range(8):
                                k = c(u)*c(v)
                                sum += k*block[u,v]*cos(((2*x+1)*u*pi)/16)*cos(((2*y+1)*v*pi)/16) #*q*Q[u,v] 
                        img_out[a+x, b+y] = np.round(1/4 * sum)
    
    img_out += 128  #addere med 128 siden vi subtraherte i kompresjon
    return img_out

def verifiser(img1, img2):
    verdi = img2 - img1
    return verdi

def transformer(img, q):
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], 
            [12, 12, 14, 19, 26, 58, 60, 55], 
            [14, 13, 16, 24, 40, 57, 69, 56], 
            [14, 17, 22, 29, 51, 87, 80, 62], 
            [18, 22, 37, 56, 68, 109, 103, 77], 
            [24, 35, 55, 64, 81, 104, 113, 92], 
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])
    kompr = bildekompresjon(img, Q, q)
    res = rekonstruksjon(kompr, Q, q)
    return res

def lagSannsynlighetsmodell(img):
    N,M = img.shape
    dict = {}
    for x in range(N):
        for y in range(M):
            key = img[x,y]
            if key in dict:
                dict[key] +=1
            else:
                dict[key] = 0
    return dict

def entropi(img):
    N,M = img.shape
    dict = lagSannsynlighetsmodell(img)
    
    H = 0
    c = 0
    for n in dict.values():
        p = n /(N*M)
        H += p*log(p)
        b = np.round(np.log2(1/p))
        c += b*p
 
    return H, c
    
def kodingsredundans():
    pass

def kompresjonsrate(ukomprimg, komprimg):
    pass

if __name__ == '__main__':
    filename = 'uio.png'
    img = imread(filename, as_gray=True)
    N, M = img.shape

    plt.subplot(2,3,1)
    plt.title("Original bilde")
    plt.imshow(img, cmap='gray')

    q = 0.1
    res1 = transformer(img, q)
    plt.subplot(2,3,2)
    plt.title("q = " + str(q))
    plt.imshow(res1, cmap='gray')

    q = 0.5
    res = transformer(img, q)
    plt.subplot(2,3,3)
    plt.title("q = " + str(q))
    plt.imshow(res, cmap='gray')

    q = 2
    res2 = transformer(img, q)
    plt.subplot(2,3,4)
    plt.title("q = " + str(q))
    plt.imshow(res2, cmap='gray')

    q = 8
    res = transformer(img, q)
    plt.subplot(2,3,5)
    plt.title("q = " + str(q))
    plt.imshow(res, cmap='gray')
    
    q = 32
    res = transformer(img, q)
    plt.subplot(2,3,6)
    plt.title("q = " + str(q))
    plt.imshow(res, cmap='gray')
    
    plt.show()

    e, c = entropi(res1)
    print("Entropi for q = 0.1: " + str(e))
    print(c)
    print("Lagringsplassen til bildet blir" + str(-e*N*M))


    CR = kompresjonsrate(img, res1)

