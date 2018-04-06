import cv2
import numpy as np
import os
import helper
from scipy.cluster.vq import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the classifier, class names, scaler, number of clusters and vocabulary 
samples = np.loadtxt('samples.data', np.float32)
responses = np.loadtxt('responses.data', np.float32)
classes_names = np.load('class_names.data.npy')
voc = np.load('voc.data.npy')

# get common parameters from helper file
num_clusters, num_neighbors = helper.getCommonParameters()

# initalize a knn classifier and train using loaded data
clf = cv2.ml.KNearest_create()
clf.train(samples, cv2.ml.ROW_SAMPLE, responses)

# path to train image files
dir_path = 'data/test'

image_paths, image_classes, class_names = helper.parseImageDirectory(dir_path)

# list to store all the descriptors
des_list = []

# helper.getPCASIFT() implments PCA-SIFT analysis on a given image
for image_path in image_paths:
	des_list.append((image_path, helper.getPCASIFT(image_path)))

# stack vertically
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
	descriptors = np.vstack((descriptors, descriptor))

# histogram of features
test_features = np.zeros((len(image_paths), num_clusters), "float32")
for i in range(len(image_paths)):
	words, distance = vq(des_list[i][1], voc)
	for w in words:
		test_features[i][w] += 1 

# classify (prediction)
retval, results, neigh_resp, dists = clf.findNearest(test_features, k=num_neighbors)

correct_count = 0
for i in range(0, len(results)):
	if(results[i] == image_classes[i]):
		correct_count += 1

accuracy = round(correct_count / len(results), 2)

cnf_matrix = confusion_matrix(y_true=image_classes, y_pred=results)

# Plot normalized confusion matrix
helper.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
					title=str(num_clusters) + ' clusters, ' + str(num_neighbors) + ' neighbors, accuracy=' + str(accuracy))