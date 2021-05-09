import numpy as np
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import skimage.measure as measure
import skimage.filters as filters
import skimage.exposure as exposure
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
        img = self.__reader.get_img()[:, :, 1]

        img = filters.gaussian(img)
        img = filters.unsharp_mask(img)
        img = exposure.equalize_hist(img)

        x_moments = []

        for i in self.__patches:
            target = 1 if expert_mask[i[2, 2, 0], i[2, 2, 1]] > 0 else 0
            y.append(target)

            tmp_img = img[i[0, 0, 0]:i[4, 4, 0]+1, i[0, 0, 1]:i[4, 4, 1]+1]
            norm_moments = measure.moments_normalized(measure.moments_central(tmp_img))

            hu_moments = measure.moments_hu(norm_moments)
            hu_moments = np.where(hu_moments != 0, hu_moments, 10**-50)

            for j in range(len(hu_moments)):
                hu_moments[j] = -1 * np.copysign(1.0, hu_moments[j]) * np.log10(abs(hu_moments[j]))

            parameters = np.append(hu_moments, np.var(tmp_img))

            x_moments.append(parameters)

        x_moments = np.array(x_moments)
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(x_moments, y, test_size=0.33)

        test = (x_test, y_test)

        resampler = RandomUnderSampler(sampling_strategy=1.0)

        train = resampler.fit_resample(x_train, y_train)

        return train, test

    def get_quarter_patches(self, quarter: int):
        if 4 < quarter < 1:
            raise Exception("Wrong quarter chosen: should be between 1 and 4")

        shp = self.__indices.shape

        start = (quarter - 1) * shp[0] // 4

        if quarter > 1:
            start -= 4

        end = quarter * shp[0] // 4

        patches = image.extract_patches_2d(self.__indices[start:end], (5, 5))
        mask = self.__reader.get_mask()
        filter_tab = []

        for i in patches:
            if mask[i[2, 2, 0], i[2, 2, 1]] == 1:
                filter_tab.append(True)
            else:
                filter_tab.append(False)

        patches = patches[filter_tab]
        patches = patches[::2]

        img = self.__reader.get_img()[:, :, 1]
        img = filters.gaussian(img)
        img = filters.unsharp_mask(img)
        img = exposure.equalize_hist(img)

        x_moments = []

        for i in patches:
            tmp_img = img[i[0, 0, 0]:i[4, 4, 0]+1, i[0, 0, 1]:i[4, 4, 1]+1]
            tmp_img = np.nan_to_num(tmp_img)
            norm_moments = measure.moments_normalized(measure.moments_central(tmp_img))

            hu_moments = measure.moments_hu(norm_moments)
            hu_moments = np.where(hu_moments != 0, hu_moments, 10 ** -50)

            for j in range(len(hu_moments)):
                hu_moments[j] = -1 * np.copysign(1.0, hu_moments[j]) * np.log10(abs(hu_moments[j]))

            parameters = np.append(hu_moments, np.var(tmp_img))

            x_moments.append(parameters)

        x_moments = np.array(x_moments)

        return patches[:, 2, 2], x_moments
