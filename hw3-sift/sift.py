import numpy as np
import cv2
import sys

print("Python version: " + sys.version)
print("OpenCV version: " + cv2.__version__)

MIN_MATCH_COUNT = 4

img1 = cv2.imread('data/image_2.jpg', cv2.IMREAD_GRAYSCALE)	# queryImage
img2 = cv2.imread('data/image_5.jpg', cv2.IMREAD_GRAYSCALE)	# trainImage

# initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# initialize a brute-force matcher and run knnMatch()
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# total number of matches before RANSAC
print('Matches before RANSAC:')
print(len(matches))

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
	if m.distance < 0.7 * n.distance:
		good.append(m)

# number of good matches
print('Number of "good" matches: ' + str(len(good)))

# sort the results based on distance
good = sorted(good, key = lambda val: val.distance)
# good = good[:15]

# filter "good" matches
if len(good) > MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

	# find homography matrix and get masks
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()

	# homography matrix
	print(M)

	h, w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)


else:
	print("Not enough matches are found - %d, %d" % (len(good),MIN_MATCH_COUNT))
	matchesMask = None



# draw inliers
draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
					singlePointColor = None,
					matchesMask = matchesMask, # draw only inliers
					flags = 2)


# number of matches after applying homography
print('Number of matches after RANSAC: ' + str(matchesMask.count(1)))

# Finding number of keypoints
# print('Number of keypoints: ' + str(len(kp1)))

# draw keypoints
# img_keypoints = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('keypoints', img_keypoints)

# draw matches
img_matches = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow('matches', img_matches)


cv2.waitKey(0)
cv2.destroyAllWindows()