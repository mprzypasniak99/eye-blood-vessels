import numpy as np
from sklearn.feature_extraction import image
from ImgObserver import ImgObserver
import skimage.measure as measure
import skimage.filters as filters
import skimage.exposure as exposure
from ImgReader import ImgReader
import random as r


class ImgPatchExtractor(ImgObserver):
    def __init__(self, reader=ImgReader):
        self.__reader = reader
        try:
            self.changed_img()
        except Exception:
            self.__indices = []
        self.__patches = np.array([])

    def __create_indices_matrix(self):
        for i in range(self.__indices.shape[0]):
            for j in range(self.__indices.shape[1]):
                self.__indices[i, j, 0] = i
                self.__indices[i, j, 1] = j

    def set_reader(self, reader: ImgReader):
        self.__reader = reader
        self.__indices = np.zeros((*self.__reader.get_img().shape[0:2], 2), dtype=int)  

    def extract_patches(self):
        self.__patches = image.extract_patches_2d(self.__indices, (5, 5), max_patches=20000)

        tmp = self.__reader.get_mask()

        filter_arr = []

        for i in self.__patches[:, 2, 2, :]:
            if tmp[i[0], i[1]] == 0:
                filter_arr.append(False)
            else:
                filter_arr.append(True)

        self.__patches = self.__patches[filter_arr]

    def get_train_sets(self):
        y_set = []
        x_set = []
        expert_mask = self.__reader.get_standard()
        img = self.__reader.get_img()[:, :, 1]
        img_mask = self.__reader.get_mask()
	
        tmp = np.mean(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img_mask[y, x] == 0:
                    img[y, x]= tmp
        img = exposure.equalize_adapthist(img)
        img = exposure.adjust_gamma(img, 0.7, 0.7)
        img = exposure.rescale_intensity(img)
	
        points = []
        while len(points) < 2250000:
            x = r.randint(2, 3501)
            y = r.randint(2, 2333)
            if img_mask[y, x] != 0:
                points.append((y, x))

        for y, x in points:
            tmp = []
            for i in range(y-2,y+3):
                for j in range(x-2,x+3):
                    tmp.append(img[i, j])
            x_set.append(tmp)
            y_set.append(expert_mask[y, x])
            
        x_set = np.array(x_set)
        y_set = np.array(y_set)

        return x_set, y_set

    def get_quarter_patches(self, quarter: int):
        if 4 < quarter < 1:
            raise Exception("Wrong quarter chosen: should be between 1 and 4")

        parameters = []

        shp = self.__indices.shape

        start = (quarter - 1) * shp[0] // 4

        if quarter > 1:
            start -= 4

        end = quarter * shp[0] // 4

        patches = image.extract_patches_2d(self.__indices[start:end], (5, 5))

        img = self.__reader.get_img()[:, :, 1]
        img_mask = self.__reader.get_mask()
        tmp = np.mean(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img_mask[y, x] == 0:
                    img[y, x]= tmp
        img = exposure.equalize_adapthist(img)
        img = exposure.adjust_gamma(img, 0.7, 0.7)
        img = exposure.rescale_intensity(img)

        for i in patches:
            parameters.append(img[i[0, 0, 0]:i[4, 4, 0] + 1, i[0, 0, 1]:i[4, 4, 1] + 1].flatten())

        parameters = np.array(parameters)
        return patches[:, 2, 2], parameters

    def changed_img(self):
        self.__indices = np.zeros((*self.__reader.get_img().shape[0:2], 2), dtype=int)
        self.__create_indices_matrix()
