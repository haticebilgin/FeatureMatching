# FeatureMatching
In order to make feature detection, interest point is needed to be found. Interest point or Feature Point is the point which is expressive in texture.
Interest point is the point at which the direction of the boundary of the object changes abruptly or intersection point between two or more edge segments.
Properties of interest point should have a well-defined position in image space or well localized. They need to be  stable under local and global perturbations
in the image domain as illumination/brightness variations, such that the interest points can be reliably computed with a high degree of repeatability. 
They also should provide efficient detection

FEATURE DETECTOR TECHNIQUES:
ORB (Oriented FAST and rotated BRIEF)
SURF (Speeded Up Robust Features)
SIFT (Scale Invariant Feature Transform)
AKAZE (Accelerated-KAZE)
ECC (Enhanced Correlation Coefficient)

FEATURE MATCHING:
Features matching or generally image matching, a part of many computer vision applications such as image registration,
camera calibration and object recognition, is the task of establishing correspondences between two images of the same scene/object.
A common approach to image matching consists of detecting a set of interest points each associated with image descriptors from image data.
Once the features and their descriptors have been extracted from two or more images, the next step is to establish some preliminary feature matches between these images.
Generally, the performance of matching methods based on interest points depends on both the properties of the underlying interest points and the choice of associated 
image descriptors. Thus, detectors and descriptors appropriate for images contents shall be used in applications. For instance, if an image contains bacteria cells,
the blob detector should be used rather than the corner detector. But, if the image is an aerial view of a city, the corner detector is suitable to find man-made structures. 
Furthermore, selecting a detector and a descriptor that addresses the image degradation is very important.

HOMOGRAPHY:
Homography is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.
Let (x1,y1) be a point in the first image and (x2,y2)} be the coordinates of the same physical point in the second image. 
