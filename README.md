# Description

## Algoritmer directory

This folder contains various Python-based image processing algorithms that improve overall quality of grayscale images.

 The scripts are versatile and can be tailored to suit specific image processing needs.

All these files can be run without arguments:
```python3 {filename}.py```

### 1 Gråtoneklassifikasjon

This alogirthm is designed to perform various operations on images, including the addition of noise, adjusting light intensity, and separating background and foreground pixels.


### 2 Gråtonetransform

This script facilitates the transformation of grayscale images, where the intensity levels incrementally increase from Imin to Imax. 

The transformation function enhances images by fine-tuning their mean and standard deviation parameters, thereby providing users with customized images with improved visual quality. 
Overall brightness  can be changed by adjusting the mean value, while modifying the standard standard deviation facilitates control over the contrast levels within the image. 

### 3 Histogramutjevning /

This script executes histogram equalization on grayscale images, generating the normalized histogram, cumulative histogram, and equalized histogram. By leveraging these operations, the script enhances the contrast and brightness of images, thereby improving their visual appeal.

### 4 Histogramtilpasning / Historgram Equalization

This script applies histogram equalization to grayscale images, resulting in improved contrast and brightness.

### 5  Requantization

This focuses on reducing the bit depth of grayscale images to enhance processing efficiency and reduce memory usage. By employing the concept of quantization, the algorithm effectively downsamples pixel values to fit within the specified bit depth, thereby optimizing image representation.

Adjust the parameter of the specific bit depth requirement as desired.

### 6 Morphology

 By leveraging the cv2 module, the script applies erosion operations on a grayscale image, reducing the boundaries of the foreground object and effectively slimming the image contours.

 Adjust the kernel size and iteration count to achieve the desired erosion effect. 

 ### 7 Pixelmanipulation

 This code showcases two distinct image manipulation techniques: Manipulating an input image, involving pixel-wise arithmetic operations and with a simplified approach utilizing element-wise multiplication.

 ## Assignment 1

**Running the code**
Simply execute the script as you would with any standard Python program. The provided code performs the specified tasks and generates the desired output.

### Oblig 1

**Task 1 - Preprocessing of Portrait Images for Face Recognition**
The objective of this task was to preprocess portrait images for subsequent facial recognition algorithms. We were provided with the image 'portrett.png,' which required contrast standardization and geometric normalization to facilitate comparisons with other images in a database. I achieved this by implementing a linear grayscale transformation to obtain an image with a mean value of 127 and a standard deviation of 64. Additionally, I standardized the geometry using an affine transformation to align the eyes and mouth with a predefined mask provided in the file 'geometrimaske.png.' My implementation included both forward and backward transformations, with experiments conducted using both nearest neighbor and bilinear interpolations. I compared the results of forward and backward mapping, as well as the different interpolation techniques.

**Task 2 - Cell Nuclei Detection**
The aim of this task was to implement the entire Canny algorithm from scratch to detect the edges of cell nuclei in the image 'cellekjerner.png.' I began by creating a general implementation of image convolution with a convolution filter. I then proceeded to implement the Canny algorithm, using a genuine Gauss filter and the symmetrical 1D operator for gradient estimation. I also applied the method described in the lecture notes for thinning the gradient magnitude and employed the 8-connection in the hysteretic thresholding step. I experimented with different values of the parameters, including sigma, Th, and Tl, to obtain an image that captured most of the edges of the cell nuclei while minimizing false positives. 

### Oblig 2

**Task 1 - Implementation of Convolution Filters in the Frequency Domain**

In this task, we were required to implement a 15×15 mean value filter using both spatial convolution and frequency domain transition. I achieved this using the conv2 function for spatial convolution and the fft2 function for the frequency domain. By comparing the results, we discovered noticeable differences between the spatial and frequency domain filtering. We also explored the computational performance of both approaches for different filter sizes and discussed the scenarios in which the frequency domain filtering is more advantageous.

**Task 2 - Lossy JPEG Compression**

For this task, we focused on implementing key aspects of lossy JPEG compression. We created a function that takes an image file name and a parameter q as input. The function calculates the approximate storage space required after JPEG compression and estimates the compression rate accordingly. We followed a series of steps including subtracting a constant from pixel intensities, performing a two-dimensional discrete cosine transform (2D DCT), and quantizing the transformed blocks. We also analyzed the trade-offs between image quality and compression ratio for different values of the parameter q, discussing the trade-offs and the suitability of specific compression rates for displaying images on regular screens.