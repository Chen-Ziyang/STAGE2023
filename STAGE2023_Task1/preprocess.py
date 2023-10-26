import numpy as np
import cv2
import os


def crop_img(img):
    ys, xs = np.where(img == np.max(img))
    y_center = int(np.mean(ys))
    if y_center - 256 < 0:
        y_center = 256
    if y_center + 256 > img.shape[0]:
        y_center = img.shape[0] - 256
    return img[y_center - 256:y_center + 256]


dataset_root = '/czy_SSD/1T/zychen/STAGE2023/STAGE_training/training_images'
# dataset_root = '/czy_SSD/1T/zychen/STAGE2023/STAGE_validation/validation_images'
lists = os.listdir(dataset_root)

for real_index in lists:
    img_lists = [p for p in os.listdir(os.path.join(dataset_root, real_index)) if p.endswith('.jpg')]
    oct_series_list = sorted(img_lists, key=lambda x: int(x.strip("_")[0]))

    oct_img_0 = cv2.imread(os.path.join(dataset_root, real_index, oct_series_list[0]), cv2.IMREAD_GRAYSCALE)
    oct_img = np.zeros((len(oct_series_list), oct_img_0.shape[0], oct_img_0.shape[1]), dtype="uint8")

    oct_crop_0 = crop_img(oct_img_0.copy())
    oct_img_crop = np.zeros((len(oct_series_list), oct_crop_0.shape[0], oct_crop_0.shape[1]), dtype="uint8")

    for k, p in enumerate(oct_series_list):
        img = cv2.imread(os.path.join(dataset_root, real_index, p), cv2.IMREAD_GRAYSCALE)
        oct_img[k] = img
        oct_img_crop[k] = crop_img(img.copy())

    print(os.path.join(dataset_root, real_index, 'img.npy'), oct_img.shape)
    print(os.path.join(dataset_root, real_index, 'img_crop.npy'), oct_img_crop.shape)

    np.save(os.path.join(dataset_root, real_index, 'img.npy'), oct_img)
    np.save(os.path.join(dataset_root, real_index, 'img_crop.npy'), oct_img_crop)
