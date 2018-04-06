from __future__ import print_function

import os
import numpy as np
import cv2

def mean_shift_segmentor(img_filename, spatial_radius, color_radius):
	src_img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
	img_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2Lab)

	# Find the peak of a color-spatial distribution
	# pyrMeanShiftFiltering(src, spatialRadius, colorRadius, max_level)
	# For 640x480 color image, it works well to set spatialRadius equal to 2 and colorRadius equal to 40
	# max_level describes how many levels of scale pyramid you want to use for segmentation
	# A max_level of 2 or 3 works well for a 640x480 color image
	dst = cv2.pyrMeanShiftFiltering(img_lab, spatial_radius, color_radius, 1)
	dst = cv2.cvtColor(dst, cv2.COLOR_Lab2BGR)

	# filename
	dst_filename = os.path.splitext(img_filename)[0] + '_meanshift_spatial_' + str(spatial_radius) + '_color_' + str(color_radius) + os.path.splitext(img_filename)[1]
	print('dst_filename: ' + dst_filename)

	cv2.imwrite(dst_filename, dst)

	return dst


def watershed_segmentor(img_filename, threshold_factor):
	src_img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

	gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# noise removal
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

	# surebackground area
	sure_bg = cv2.dilate(opening, kernel, iterations=3)
	
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, threshold_factor * dist_transform.max(), 255, 0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)
	
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers + 1
	# Now, mark the region of unknown with zero
	markers[unknown == 255] = 0

	dst = src_img.copy()

	markers = cv2.watershed(dst, markers)
	dst[markers == -1] = [255, 0, 0]

	# filename
	dst_filename = os.path.splitext(img_filename)[0] + 'watershed_threshold_' + str(threshold_factor) + os.path.splitext(img_filename)[1]
	print('dst_filename: ' + dst_filename)

	cv2.imwrite(dst_filename, dst)

	return dst



file_names = ['300091.jpg', '101085.jpg', '253027.jpg']
threshold_params = [x * 0.1 for x in range(0, 10)]

for file_name in file_names:
	for spatial_radius in range(1, 6, 1):
		for color_radius in range(10, 60, 10):
			mean_shift_segmentor(file_name, spatial_radius, color_radius)

	for threshold in threshold_params:
		watershed_segmentor(file_name, threshold)