import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import confusion_matrix

fea_det = cv2.xfeatures2d.SIFT_create()

# number of clusters
num_clusters = 150
num_neighbors = 12

# return common parameters for both train/classify
def getCommonParameters():
	return num_clusters, num_neighbors

# directories are separated by labels
def parseImageDirectory(dir_path):
	class_names = os.listdir(dir_path)

	# image paths
	image_paths = []

	# image classes corresponding to image_paths list
	image_classes = []

	# class ID begins at 0, increment by when navigating to a new directory
	class_id = 0

	# iterate over each folder
	for class_name in class_names: 
		dir = os.path.join(dir_path, class_name)
		listed_image_paths = [os.path.join(dir, f) for f in os.listdir(dir)]
		image_paths += listed_image_paths

		# add class labels for each image inside the same directory
		image_classes += [class_id] * len(listed_image_paths)

		class_id += 1

	return image_paths, image_classes, class_names


def getPCASIFT(image_path):
	img = cv2.imread(image_path)

	# find keypoints and descriptors with SIFT
	kpts, des = fea_det.detectAndCompute(img, None)

	# perform PCA with SciKit module
	# note that the module is imported as sklearnPCA instead of PCA to avoid conflicts with other modules
	sklearn_pca = sklearnPCA(n_components=20)
	pca_des = sklearn_pca.fit_transform(des)

	return pca_des


def plot_confusion_matrix(cm, classes,
							normalize=False,
							title='Confusion matrix',
							cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()