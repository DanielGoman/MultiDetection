import numpy as np
import cv2
from matplotlib import pyplot as plt
import tifffile

from scipy.signal import wiener


class Steps:
    def __init__(self):
        self.steps = []

    def add(self, step):
        self.steps.append(step)

    def run(self, image):
        curr_image = image.copy()
        for step in self.steps:
            curr_image = step(curr_image)

        return curr_image


def run_steps(image_path: str):
    image = tifffile.imread(image_path)

    steps = Steps()
    steps.add(min_max_normalization)
    # steps.add(histogram_equalization)
    steps.add(wiener_filter)

    out_image = steps.run(image=image)

    step_names = [step.__name__ for step in steps.steps]
    cv2.imwrite(fr'C:\Users\roey\PycharmProjects\MultiDetection\out\{step_names}.tif', out_image)
    show_image(image=out_image)


def show_image(image: np.ndarray):
    # plt.imshow(out_image)
    # plt.show()
    reshape_sizes = (1500, 1000)

    image = cv2.resize(image, reshape_sizes)
    cv2.imshow('image', image)
    cv2.waitKey()


def min_max_normalization(image: np.ndarray):
    normalized_image = 255 * ((image - np.min(image)) / (np.max(image) - np.min(image)))
    return normalized_image.astype(np.uint8)


def histogram_equalization(image: np.ndarray):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf.max()
    out_vals = (np.max(256) * cdf).astype(np.uint8)

    out_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out_image[i][j] = out_vals[image[i][j]]

    # plt.hist(out_image.flatten(), 256, [0, 256])
    # plt.show()

    return out_image.astype(np.uint8)


def wiener_filter(image: np.ndarray):
    filtered_image = wiener(image, (50, 40))
    return filtered_image


def test(image_path: str):
    image = tifffile.imread(image_path)
    histogram_equalization(image=image)


if __name__ == '__main__':
    image_path = r'C:\Users\roey\PycharmProjects\MultiDetection\data\IMG_0065_7.tif'
    run_steps(image_path=image_path)
    # test(image_path=image_path)
