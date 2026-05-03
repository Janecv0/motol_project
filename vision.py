"""Blob detection helpers used by the camera preview and experiment logging."""

from __future__ import annotations

import cv2


def get_blob_detector(min_area=30, max_area=600):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)


def find_marker_centers(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    keypoints = detector.detect(gray_blur)
    centers = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    return centers
