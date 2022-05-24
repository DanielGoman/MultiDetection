import cv2
import numpy as np

from typing import List


def register_two_channels(ch1: np.ndarray, ch2: np.ndarray):
    height, width = ch2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(ch1, None)
    kp2, d2 = orb_detector.detectAndCompute(ch2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_ch2 = cv2.warpPerspective(ch1, homography, (width, height))

    return transformed_ch2


def register_all_channels(all_channels: List[np.ndarray]):
    reference_channel = all_channels[0]
    shape = reference_channel.shape
    num_channels = len(all_channels)
    aligned_channels = np.empty((num_channels, *shape))
    aligned_channels[0] = reference_channel

    for i in range(1, len(all_channels)):
        curr_channel = all_channels[i]
        transformed_channel = register_two_channels(ch1=reference_channel, ch2=curr_channel)
        aligned_channels[i] = transformed_channel
        print(f'Registered channels 1 and {i + 1}')

    all_channels_image = np.transpose(aligned_channels, (1, 2, 0))

    return all_channels_image
