import skimage.io as skio
import numpy as np
import core as c
import matplotlib.pyplot as plt
    
# TODO import file
data = skio.imread('data/Chi_10m_12bands' + '.tif', plugin="tifffile")
# NOTE clearning NaN
data, _, _ = c.cleaning_data(data)
# print(data.shape)

# TODO select band to plot
band_number = 8 
data = data[:, :, band_number-1]
max_num, min_num = c.clip(data, 5)
# c.plot_imshow(data, min_num, max_num, 'band: ' + str(band_number),\
#                 'band_' + str(band_number))

# TODO k-means
number_of_classes = 3
kmeans = c.compute_kmeans(data, number_of_classes)
c.plot_kmeans(kmeans, 'k-means', 'kmeans')

min_pixel = 1
max_pixel = 1000
image_seg = c.image_segmentation(kmeans, min_pixel, max_pixel)
c.plot_kmeans(image_seg, 'k-means + image segmentation', 'image_seg')