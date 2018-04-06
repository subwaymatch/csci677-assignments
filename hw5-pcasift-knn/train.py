import cv2
import helper
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# get common parameters from helper file
num_clusters, num_neighbors = helper.getCommonParameters()

# path to train image files
dir_path = 'data/train'

image_paths, image_classes, class_names = helper.parseImageDirectory(dir_path)

print('class_names')
print(class_names)

# list to store all the descriptors
des_list = []

# helper.getPCASIFT() implments PCA-SIFT analysis on a given image
for image_path in image_paths:
	des_list.append((image_path, helper.getPCASIFT(image_path)))

# stack vertically
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
	descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors

# k-means clustering with 'num_clusters'
voc, variance = kmeans(descriptors, num_clusters, 1)

# histogram of features
im_features = np.zeros((len(image_paths), num_clusters), "float32")
for i in range(len(image_paths)):
	words, distance = vq(des_list[i][1], voc)
	for w in words:
		im_features[i][w] += 1

# save trained data to files
np.savetxt("samples.data", im_features)
np.savetxt("responses.data", np.array(image_classes))
np.save("class_names.data", class_names)
np.save("voc.data", voc)




