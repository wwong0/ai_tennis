import cv2
import numpy as np

w, h = 512, 512
src = np.array(
    [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
dst = np.array(
    [[300, 350], [800, 300], [900, 923], [161, 923]], dtype=np.float32)

m = cv2.getPerspectiveTransform(src, dst)
print(m)
result = cv2.perspectiveTransform(src[None, :, :], m)
print(result)

