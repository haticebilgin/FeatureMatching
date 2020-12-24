import numpy as np
import cv2
from PIL._imaging import display
from cv2.xfeatures2d import matchGMS
import glob
import operator

# LOAD IMAGES ---------------------------------------------------
size_ratio = 0.25


def feature_matching(img1, img2):
    # ORB DESCRIPTOR------------------------------------------------------------------------------
    orb = cv2.ORB_create(nfeatures=10000)
    orb.setFastThreshold(0)  # This enables matching in weakly textured environments
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # GMS MATCHER --------------------------------------------------------------------------------
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)  # list of DMatch objects
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=True,
                           thresholdFactor=3)

    output = cv2.drawMatches(img1, kp1, img2, kp2, matches_gms[:20], None, flags=2)
    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    # cv2.imshow("result", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    height, width = img1.shape[:2]
    if len(matches_gms) >= 4:
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5)
        # print("homography matrix: \n", matrix)
        pts = np.float32([[0, 0], [0, height], [height, width], [width, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        # print("points: ", dst)
        # A = dst.flatten()

    else:
        print("no matches sorry")

    return len(matches_gms)


if __name__ == "__main__":
    images = []
    images_wf = []
    images = [cv2.imread(file, 0) for file in glob.glob("C:\\Users\\Lenovo\\Desktop\\matchingPics\\mf*.jpg")]
    images[0] = images[0][1945:2945, 95:1695]
    #info restricted 

    images_wf = [cv2.imread(file, 0) for file in
                 glob.glob("C:\\Users\\Lenovo\\Desktop\\matchingPics\\wfimages\\wf*.png")]
    print(images_wf[0].shape)
    img1 = images[5]
    img2 = images_wf[1]
    size_of_wf = len(images_wf)
    size_of_mf = len(images)

    matches_total = []
    best_matches = []
    for i in range(0, 12):
        matches_total.append([])
        for j in range(0, 9):
            matches_total[i].append(feature_matching(images[i], images_wf[j]))
        max_val = max(matches_total[i])
        max_index = matches_total[i].index(max_val)
        best_matches.append((max_val, max_index))

    print(best_matches)

