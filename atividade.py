import matplotlib.pyplot as mp
import numpy as np

def imread(filename):
    im = mp.imread(filename)
    if (im.dtype == "float32"):
        im = np.uint8(255*im)
    if (len(im.shape) >= 3 and im.shape[2] > 3):
        im = im[:, :, 0:3]
    return im

def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper")
    plot.set_interpolation('nearest')
    mp.show()

# Q1

def nchannels(im):
    if len(im.shape) == 2:
        return 1

    return im.shape[2]

# Q2

def size(im):
    return [im.shape[1], im.shape[0]]

# Q3

def rgb2gray(im):
    if nchannels(im) == 1:
        return im
    
    imggray = np.empty([im.shape[0], im.shape[1]])
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            imggray[i][j] = im[i][j][0] * 0.299 + im[i][j][1] * 0.587 + im[i][j][2] * 0.114

    return np.array(imggray, dtype = np.uint8)

# Q4

def imreadgray(filename):
    img = imread(filename)
    imggray = rgb2gray(img)

    return imggray

# Q5

def thresh(im, valor):
    imgthresh = rgb2gray(im)
    for i in range(0, im.shape[0]): 
        for j in range(0, im.shape[1]):
            if imgthresh[i][j] >= valor:
                imgthresh[i][j] = 255
            else:
                imgthresh[i][j] = 0

    return imgthresh

# Q6

def negative(im):
    imgnegative = 1 - im

    return imgnegative

# Q7

def contrast(im, r, m):
    g = r * (im - m) + m

    return np.array(g, dtype = np.uint8)

# whenn
