import os
import cv2

from utils.registration import register_all_channels

DATA_DIR = os.path.abspath('../data')
images_paths = ['IMG_0333_1.tif', 'IMG_0333_2.tif', 'IMG_0333_3.tif', 'IMG_0333_4.tif', 'IMG_0333_5.tif']


def main():
    images = []
    images_full_paths = [os.path.join(DATA_DIR, image_path) for image_path in images_paths]
    for image_path in images_full_paths:
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        print(f'Read {image_path}')
        images.append(image)

    register_all_channels(all_channels=images)


if __name__ == '__main__':
    main()
