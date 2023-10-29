import matplotlib.pyplot as plt
import imageio as io
import numpy as np
import math

bit = 8;

img = io.imread('bilder/lena.png', as_gray=True)
plt.imshow(img, cmap='gray')
f_requantized = math.floor(img/2**bit);
plt.imagesc(f_requantized, [0, 2**bit-1])
plt.colorbar()