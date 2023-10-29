import matplotlib.pyplot as plt
import numpy as np

# Første del:
filename = 'bilder/mona.png'
f = plt.imread(filename)
plt.imshow(f, cmap='gray')

# Andre del:
N,M = f.shape
f_out = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        f_out[i,j] = (f[i,j] - f[i-1,j]) * 1000 + 128

plt.figure()
plt.imshow(f_out, cmap='gray', vmin=0, vmax=255, aspect='auto')
plt.title('f_out')

# Alternativ andre del med litt mindre kodelinjer som gjør akkurat det samme:
f_out_alt = 0.5*f

plt.figure()
plt.imshow(f_out_alt,cmap='gray',vmin=0,vmax=255,aspect='auto')
plt.title('f_out_alt')

plt.show()