import math
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import sympy as sp

def graatonetransf(filename, mu_t, sigma_t):
    img = imread(filename, as_gray=True)
    plt.imshow(img, cmap = 'gray')
    N,M = img.shape
    img_out = np.zeros((N,M))

    sigma = np.std(img)
    mean = np.mean(img)
    a = np.sqrt((sigma_t**2)/(sigma**2))
    b = mu_t - a * mean

    for i in range(N):
        for j in range(M):
            img_out[i,j] = a * img[i,j] + b

    print("Naa vises mellom-resultat-bildet etter graatonetransformasjonen:)")
    vindu = np.hstack((img, img_out))
    plt.imshow(vindu, cmap = 'gray', vmin=0, vmax=255)
    plt.title("Graatonetransform")
    plt.show()

# x′= a0x + a1y + a2
# y′= b0x + b1y + b2

def forwardmapping(filename, maskfile):
    img = imread(filename, as_gray=True)
    plt.imshow(img, cmap = 'gray')
    img2 = imread(maskfile, as_gray=True)

    N, M = img.shape
    N2,M2 = img2.shape
    img_out = np.zeros((N2,M2))

    T = findT()

    for x in range(N):
        for y in range(M):
            x_out = round(T[0]*x + T[1]*y + T[2])
            y_out = round(T[3]*x + T[4]*y + T[5])
            x = int(x)
            y = int(y)
            if x_out in range(N2) and y_out in range(M2):
                img_out[x_out,y_out] = img[x,y]
   
    print("Viser resultatbilder for forlengstransformasjon")
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.08)
    plt.title("Forlengsmapping")
    plt.show()
    return img_out

def backwardmapping_nabo(filename, maskfile):
    img = imread(filename, as_gray=True)
    plt.imshow(img, cmap = 'gray')
    img2 = imread(maskfile, as_gray=True)
    N,M = img.shape
    N2,M2 = img2.shape
    img_out = np.zeros((N2,M2))

    T = findT()
    A = np.array([[T[0], T[1], T[2]], 
         [T[3], T[4], T[5]], 
         [0, 0, 1]])
    T = np.linalg.inv(A)

    for x_out in range(N2):
        for y_out in range(M2):
            x = round(T[0][0]*x_out + T[0][1]*y_out + T[0][2])
            y = round(T[1][0]*x_out + T[1][1]*y_out + T[1][2])
            x_out = int(x_out)
            y_out = int(y_out)
            if x in range(N) and y in range(M):
                img_out[x_out,y_out] = img[x,y]
            else:
                img_out[x_out, y_out] = 0

    print("Viser resultatbilder for baklengstransformasjon med naermeste nabo")
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.15)
    plt.title("Baklengsmapping med nabo interpolasjon")
    plt.show()
    return img_out

def backwardmapping_bilinear(filename, maskfile):
    img = imread(filename, as_gray=True)
    plt.imshow(img, cmap = 'gray')
    img2 = imread(maskfile, as_gray=True)
    N,M = img.shape
    N2,M2 = img2.shape
    img_out = np.zeros((N2,M2))

    T = findT()
    A = np.array([[T[0], T[1], T[2]], 
         [T[3], T[4], T[5]], 
         [0, 0, 1]])
    T = np.linalg.inv(A)

    for x in range(N2):
        for y in range(M2):
            x_mark = T[0][0]*x+T[0][1]*y+T[0][2]
            y_mark = T[1][0]*x+T[1][1]*y+T[1][2]

            x0 = int(np.floor(x_mark)); y0 = int(np.floor(y_mark))
            x1 = int(np.ceil(x_mark)); y1 = int(np.ceil(y_mark))

            Δx = x_mark - x0
            Δy = y_mark - y0

            p = img[x0,y0] + Δx*(img[x1,y0] - img[x0, y0])
            q = img[x0,y1] + Δx*(img[x1,y1] - img[x0, y1])

            img_out[x,y] = p + Δy*(q-p)
    
    print("Viser resultatbilder for baklengstransformasjon med bilinear interpolasjon")
    plt.imshow(img_out, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255, alpha = 0.15)
    plt.title("Baklengsmapping med bilineaer interpolasjon")
    plt.show()
    return img_out

def findT():
        a = np.array([[88, 84, 1, 0, 0, 0], 
                     [0, 0, 0, 88, 84, 1], 
                     [68, 120, 1, 0, 0, 0], 
                     [0, 0, 0, 68, 120, 1],
                     [109, 130, 1, 0, 0, 0], 
                     [0, 0, 0, 109, 130, 1]])
        b = np.array([257, 170, 257, 340, 440, 256])
        x = np.linalg.solve(a,b)
        return x

if __name__ == '__main__':
    print("Oppgave 1: Preprosessering av portrett for ansiktsgjennkjenning")
    graatonetransf('portrett.png', 127, 64)
    f = forwardmapping('portrett.png', 'geometrimaske.png')
    b1 = backwardmapping_nabo('portrett.png', 'geometrimaske.png')
    b2 = backwardmapping_bilinear('portrett.png', 'geometrimaske.png')
    plt.imshow(np.hstack((f, b1, b2)), cmap = "gray", vmin=0, vmax=255)

    plt.show()

