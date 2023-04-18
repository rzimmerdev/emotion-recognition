import numpy as np
from cv2 import cv2


def dense_optical_flow(x):  # Gunnar-Farneback
    if len(x) <= 0:
        return x

    mask = np.zeros_like(x)

    curr_mask = np.zeros_like(x[0])
    curr_mask[..., 1] = 255

    prev = np.zeros_like(x[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for idx, frame in enumerate(x):
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(curr, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        curr_mask[..., 0] = angle * 90 / np.pi
        curr_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        mask[idx] = curr_mask

    return mask
