import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Images and intrinsics
img1 = cv2.imread('data/0000.png',0)
img2 = cv2.imread('data/0001.png',0)
K = np.array([[2759.48,     0.,    1520.69], [0., 2764.16, 1006.81], [0.,   0.,   1.]], np.float32)

#Extract KAZE features
kaze = cv2.KAZE_create()
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2, None)
# create BFMatcher object and match discriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Draw matches
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
plt.imshow(img3),plt.show()

#Select good matches
pts1 = []
pts2 = []
for m,n in matches:
	if m.distance<0.75*n.distance:
		pts1.append([kp1[m.queryIdx].pt])
		pts2.append([kp2[m.trainIdx].pt])

pts1 = np.float32(pts1).reshape(-1,1,2)
pts2 = np.float32(pts2).reshape(-1,1,2)

# Normalize for Esential Matrix calaculation
pts1_norm = cv2.undistortPoints(pts1, cameraMatrix=K, distCoeffs=None)
pts2_norm = cv2.undistortPoints(pts2, cameraMatrix=K, distCoeffs=None)
E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0.0,0.0), method=cv2.RANSAC, prob=0.999, threshold=0.001)

#Select inliers after RANSAC
mask1 = [bool(val) for val in mask]
pts1_norm_inliers = pts1_norm[mask1,:]
pts2_norm_inliers = pts2_norm[mask1,:]

#Triangulate 3D points
points, R, t, mask2 = cv2.recoverPose(E, pts1_norm_inliers, pts2_norm_inliers)
M1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
M2 = np.hstack((R, t))
point_4d_hom = cv2.triangulatePoints(M1, M2, pts1_norm_inliers, pts2_norm_inliers)
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T

#Lets visualize
fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim3d(0, 10)                   
ax.set_ylim3d(-5, 5)                    
ax.set_xlim3d(-5, 5)                    
ax.scatter(point_3d[:,0], point_3d[:,1], point_3d[:,2], s=3)
plt.show()

