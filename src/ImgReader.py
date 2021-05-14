import numpy as np
import skimage.io as io
import skimage.color as color
import matplotlib.pyplot as plt
from ImgObserver import *


class ImgReader:
    def __init__(self):
        self.__img = np.array([])
        self.__img_mask = np.array([])
        self.__img_golden_standard = np.array([])
        self.__observers = []

    def load_image(self, img_name: str, images_folder_path: str):
        try:
            self.__img = io.imread(images_folder_path + "images/" + img_name + ".jpg")
        except FileNotFoundError:
            self.__img = io.imread(images_folder_path + "images/" + img_name + ".JPG")
        self.__img_mask = color.rgb2gray(plt.imread(images_folder_path + "mask/" + img_name + "_mask.tif"))
        self.__img_golden_standard = color.rgb2gray(plt.imread(images_folder_path + "manual1/" + img_name + ".tif"))

        for obs in self.__observers:
            obs.changed_img()

    def get_img(self) -> np.ndarray:
        if self.__img.size == 0:
            raise Exception("No read image in memory")
        return np.copy(self.__img)

    def get_mask(self) -> np.ndarray:
        if self.__img_mask.size == 0:
            raise Exception("No read image in memory")
        return np.copy(self.__img_mask)

    def get_standard(self) -> np.ndarray:
        if self.__img_golden_standard.size == 0:
            raise Exception("No read image in memory")
        return np.copy(self.__img_golden_standard)

    def add_observer(self, obs: ImgObserver):
        self.__observers.append(obs)