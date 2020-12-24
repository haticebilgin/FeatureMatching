import numpy as np
import cv2
# ORB + BRUTE FORCE Matcher + Homography
# -------------------------------------------------------------------------------------------
img1 = cv2.imread('C:\\Users\\Lenovo\\Desktop\\x.JPG', 0)
img1=cv2.Canny(img1)
img1 = img1[1850:2850, 2355:3955]

img2 = cv2.imread('C:\\Users\\Lenovo\\Desktop\\x.JPG', 0)

img2=img2[3000:4000, 0:3000]
img2=cv2.Canny(img2)
#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#img2 = cv2.filter2D(img2, -1, kernel)
size_ratio = 0.25
img2 = cv2.resize(img2, (0, 0), fx=size_ratio, fy=size_ratio, interpolation=cv2.INTER_NEAREST)

cv2.namedWindow("wf image", cv2.WINDOW_NORMAL)
cv2.imshow("wf image", img2)
# ORB  --------------------------------------------------------------------------------------
orb = cv2.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

for d in des1:
    print(d)

# BRUTEFORCE MATCHER--------------------------------------------------------------------------
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(des1, des2, None)
matches = sorted(matches, key=lambda x: x.distance)
print("match sayisi:", len(matches))
# -------------------------------------------------------------------------------------------
# Remove not so good matches
# GOOD_MATCH_PERCENT = 5
# numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
# matches = matches[:numGoodMatches]
# -------------------------------------------------------------------------------------------
# HOMOGRAPHY
# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt  # it gives index of the descriptor
    points2[i, :] = kp2[match.trainIdx].pt
print("Points 2: ", points2)

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
print("h: ", h)
im1 = cv2.imread('C:\\Users\\Lenovo\\Desktop\\x.JPG')
im1 = im1[1850:2850, 2355:3955]

im2 = cv2.imread('C:\\Users\\Lenovo\\Desktop\\x.JPG')
im2 = cv2.resize(im2, (0, 0), fx=size_ratio, fy=size_ratio, interpolation=cv2.INTER_NEAREST)
height, width, channels = im2.shape

result = cv2.warpPerspective(im1, h, (height, width))
result1 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], None, flags=2)
# SHOW RESULTS ----------------------------------------------------------------------------------
cv2.namedWindow("result1", cv2.WINDOW_NORMAL)
cv2.imshow('result1', result1)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
