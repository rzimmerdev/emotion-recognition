import numpy as np
import cv2


def dense_optical_flow(frames):  # Gunnar-Farneback
    if len(frames) <= 0:
        return frames

    frames = np.array(np.transpose(frames, (0, 2, 3, 1)), dtype=np.float32)

    mask = np.zeros(frames.shape[-3:])
    mask[..., 1] = 255
    masks = []

    prev = np.zeros(frames.shape[-3:-1])

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 90 / np.pi
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        masks.append(mask.copy())
        prev = gray

    return np.transpose(np.array(masks), (0, 3, 1, 2))


class HaarcascadeClipper:
    def __init__(self, model_weights_path="../data/weights/haarcascade_frontalface_default.xml"):
        self.model = cv2.CascadeClassifier(model_weights_path)

    def detect_faces(self, image) -> list[np.ndarray]:
        return self.model.detectMultiScale(image, 1.1, 4)  # Grayscale image, 1.1 scale factor, 4 MinNeighbours

    def crop_face(self, image, shape):
        face_loc: tuple = self.detect_faces(image)[0]

        x, y, w, h = face_loc

        pad_x = shape[0] // 2
        pad_y = shape[1] // 2

        x = max(0, x - pad_x)
        w = min(image.shape[0] - x, w + (shape[0] - pad_x))

        y = max(0, y - pad_y)
        h = min(image.shape[1] - y, h + (shape[1] - pad_y))

        cropped_image = image[y:y + h, x:x + w]

        return cropped_image
