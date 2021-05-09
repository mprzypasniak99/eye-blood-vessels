import numpy as np
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import skimage.measure as measure
from ImgReader import ImgReader


class ImgPatchExtractor:
    def __init__(self, reader=ImgReader):
        self.__reader = reader
        self.__indices = np.zeros((*self.__reader.get_img().shape[0:2], 2), dtype=int)
        self.__create_indices_matrix()
        self.__patches = np.array([])

    def __create_indices_matrix(self):
        tmp = np.indices(self.__indices.shape[0:2], sparse=True)
        for i in range(self.__indices.shape[0]):
            for j in range(self.__indices.shape[1]):
                self.__indices[i, j, 0] = tmp[0][i, 0]
                self.__indices[i, j, 1] = tmp[1][0, j]

    def set_reader(self, reader: ImgReader):
        self.__reader = reader
        self.__indices = np.indices(self.__reader.get_img().shape)

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

    def get_train_test_sets(self):
        y = []
        expert_mask = self.__reader.get_standard()
        image = self.__reader.get_img()[:, :, 1]

        x_moments = []

        for i in self.__patches:
            y.append(expert_mask[i[2, 2, 0], i[2, 2, 1]])

            img = image[i[0, 0, 0]:i[4, 4, 0], i[0, 0, 1]:i[4, 4, 1]]
            norm_moments = measure.moments_normalized(measure.moments_central(img))

            hu_moments = measure.moments_hu(norm_moments)

            x_moments.append(hu_moments)

        x_moments = np.array(x_moments)
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(x_moments, y, test_size=0.33)

        test = (x_test, y_test)

        resampler = RandomUnderSampler()

        train = resampler.fit_resample(x_train, y_train)

        return train, test
