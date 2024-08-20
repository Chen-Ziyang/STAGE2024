import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


def crop_img(img):
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(img.copy(), kernel, iterations=3)
    ys, xs = np.where(eroded_image.mean(-1) > 10)
    min_y, max_y, min_x, max_x = int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs))
    return img[min_y:max_y, min_x:max_x, :]


def crop_oct(img):
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(img.copy(), kernel, iterations=3)

    ys, xs = np.where(eroded_image == np.max(eroded_image))
    y_center = int(np.mean(ys))
    if y_center - 256 < 0:
        y_center = 256
    if y_center + 256 > img.shape[0]:
        y_center = img.shape[0] - 256
    return img[y_center - 256:y_center + 256]


def process_image(args):
    k, p, dataset_root, real_index = args
    img_path = os.path.join(dataset_root, real_index, p)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cropped_img = crop_oct(img.copy())
    return k, img, cropped_img


# dataset_root = '/media/user/MICCAI2024-STAGE2/STAGE_training/training_images'
dataset_root = '/media/user/MICCAI2024-STAGE2/STAGE_validation/validation_images'
lists = os.listdir(dataset_root)

for real_index in lists:
    # Fundus Preprocess
    fundus_img = cv2.imread(os.path.join(dataset_root, real_index, str(real_index)+'.jpg'))
    fundus_img_crop = crop_img(fundus_img.copy())
    np.save(os.path.join(dataset_root, real_index, 'fundus.npy'), fundus_img)
    np.save(os.path.join(dataset_root, real_index, 'fundus_crop.npy'), fundus_img_crop)
    print(os.path.join(dataset_root, real_index, 'fundus.npy'), fundus_img.shape)
    print(os.path.join(dataset_root, real_index, 'fundus_crop.npy'), fundus_img_crop.shape)

    # OCT Preprocess
    img_lists = [p for p in os.listdir(os.path.join(dataset_root, real_index)) if p.endswith('image.jpg')]
    oct_series_list = sorted(img_lists, key=lambda x: int(x.strip("_")[0]))

    oct_img_0 = cv2.imread(os.path.join(dataset_root, real_index, oct_series_list[0]), cv2.IMREAD_GRAYSCALE)
    oct_img = np.zeros((len(oct_series_list), oct_img_0.shape[0], oct_img_0.shape[1]), dtype="uint8")

    oct_crop_0 = crop_oct(oct_img_0.copy())
    oct_img_crop = np.zeros((len(oct_series_list), oct_crop_0.shape[0], oct_crop_0.shape[1]), dtype="uint8")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_image, [(k, p, dataset_root, real_index) for k, p in enumerate(oct_series_list)])

    for k, img, cropped_img in results:
        oct_img[k] = img
        oct_img_crop[k] = cropped_img

    np.save(os.path.join(dataset_root, real_index, 'img.npy'), oct_img)
    np.save(os.path.join(dataset_root, real_index, 'img_crop.npy'), oct_img_crop)

    print(os.path.join(dataset_root, real_index, 'img.npy'), oct_img.shape)
    print(os.path.join(dataset_root, real_index, 'img_crop.npy'), oct_img_crop.shape)
    print('\n')
