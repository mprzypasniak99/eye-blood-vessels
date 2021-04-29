import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as filters
import skimage.exposure as exposure
import imblearn.metrics as metrics
from ImgReader import ImgReader


class ImgProcessor:
    def __init__(self, reader=ImgReader):
        self.__processed_img = np.array([])
        self.__reader = reader

    def set_reader(self, reader: ImgReader):
        self.__reader = reader

    def show_img(self):
        io.imshow(self.__processed_img)
        plt.show()

    def process(self):
        try:
            self.__processed_img = self.__reader.get_img()[:, :, 1]
        except Exception:
            print("No image read by reader or reader not set")
            return

        self.__processed_img = filters.gaussian(self.__processed_img)
        self.__processed_img = filters.unsharp_mask(self.__processed_img)
        self.__processed_img = exposure.equalize_hist(self.__processed_img)

        self.__processed_img = filters.frangi(self.__processed_img)

        self.__processed_img = exposure.rescale_intensity(self.__processed_img, out_range=(0.0, 1.0))
        self.__processed_img = self.__round()
        self.__processed_img = exposure.rescale_intensity(self.__processed_img, out_range=(0.0, 1.0))
        self.__use_mask()

    def __use_mask(self):
        mask = self.__reader.get_mask()
        self.__processed_img = np.multiply(self.__processed_img, mask)

    def __round(self):
        shape = self.__processed_img.shape
        new = np.zeros(shape, dtype=int)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.__processed_img[i, j] > 0.05:
                    new[i, j] = 1
                else:
                    new[i, j] = 0

        return new

    def show_metrics(self):
        standard = exposure.rescale_intensity(self.__reader.get_standard(), out_range=(0.0, 1.0))
        print(metrics.classification_report_imbalanced(standard.flatten(), self.__processed_img.flatten()))
