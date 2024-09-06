# custom dataset
# keyboard vs. mouse classification

import os
import glob
import numpy as np
import random
import cv2

mouse_list = []
keyboard_list = []

class DataLoader():
    def __init__(self, dataset_name = 'mouse_vs_keyboard', test=False, shuffle=False, normalize=True):
        # Initialize variables

        path_to_mouse_datasets = os.path.join('..', '004 custom dataset', 'mouse_vs_keyboard', 'mouse_image', '*.jpg')
        path_to_keyboard_datasets = os.path.join('..', '004 custom dataset', 'mouse_vs_keyboard', 'keyboard_image',
                                                 '*.jpg')
        mouse_files = glob.glob(path_to_mouse_datasets)
        keyboard_files = glob.glob(path_to_keyboard_datasets)

        self.mouse_image = np.empty((len(self.mouse_files), 256, 256))
        self.keyboard_image = np.empty((len(self.keyboard_files), 256, 256))

        self.mouse_data_label = np.zeros(len(mouse_files))
        self.keyboard_data_label = np.ones(len(keyboard_files))


        # Settings
        self.dataset_name = dataset_name
        self.normalize_flag = normalize
        self.shuffle = shuffle

    def normalize(self, flag, arr):
        if self.normalize_flag:
            arr = arr - arr.mean()
            arr = arr / abs(arr).max()

        return arr

    def create(self):
        mouse_jpgs = sorted(glob.glob(os.path.join('..', '004 custom dataset', 'mouse_vs_keyboard',
                                                   'mouse_image', '*.jpg')))
        keyboard_jpgs = sorted(glob.glob(os.path.join('..', '004 custom dataset', 'mouse_vs_keyboard',
                                                   'keyboard_image', '*.jpg')))

        if self.shuffle:
            random.shuffle(mouse_jpgs)
            random.shuffle(keyboard_jpgs)

        jpg_idx = 0
        for _jpg in mouse_jpgs:
            self.mouse_image[jpg_idx] = cv2.imread(_jpg, cv2.IMREAD_GRAYSCALE)
            jpg_idx += 1

        jpg_idx = 0
        for _jpg in keyboard_jpgs:
            self.keyboard_image[jpg_idx] = cv2.imread(_jpg, cv2.IMREAD_GRAYSCALE)
            jpg_idx += 1

        self.mouse_image = self.normalize("mouse", np.asarray(self.mouse_image))
        self.keyboard_image = self.normalize("keyboard", np.asarray(self.keyboard_image))

        np.savez(os.path.join('..', self.dataset_name+'.npz'), mouse = self.mouse_image,
                 keyboard = self.keyboard_image)
if __name__ == "__main__":
    # create instance as dl
    dl = DataLoader()
    dl.create()
    dl.read()