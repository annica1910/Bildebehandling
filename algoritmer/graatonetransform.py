import math
import matplotlib.pyplot as plt
import numpy as np

#Et histogram for et bilde N x M der I trinnvis øker fra Imin til Imax
#N = kol, M = rad, k = bits, 
def graatonetransform(filename, mu_t, sigma_t):
    img = plt.imread(filename)
    plt.imshow(img, cmap = "gray")
    N,M = img.shape
    plt.figure()
    plt.hist(img)
    img_out = np.zeros((N,M))

    a = sigma_t/np.std(img)
    b = mu_t - a * np.mean(img)

    for i in range(N):
        for j in range(M):
            img_out[i,j] = a * math.log(img[i,j]) + b

    plt.figure()
    plt.hist(img_out)
    plt.title("Histogram of " + filename)

    plt.figure()
    plt.imshow(img_out)

graatonetransform('mona.png', 250, 80)
# graatonetransform('lena.png', 8)
