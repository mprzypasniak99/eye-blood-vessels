from ImgPatchExtractor import ImgPatchExtractor
from ImgReader import ImgReader
from sklearn import neighbors
from imblearn import metrics
import numpy as np
import os


class ImgMachineLearning:
    def __init__(self, reader: ImgReader):
        self.__classifier = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
        self.__reader = reader
        self.__patch = ImgPatchExtractor(reader)

    @staticmethod
    def __get_files_list():
        lst = []
        for file in os.listdir("../img/images"):
            if file.endswith(".jpg") or file.endswith(".JPG"):
                lst.append(file[:-4])

        return lst

    def train(self):
        files = self.__get_files_list()

        all_train = [[], []]
        all_test = [[], []]

        for file in files[:10]:
            self.__reader.load_image(file, "../img/")

            self.__patch.extract_patches()

            train, test = self.__patch.get_train_test_sets()

            all_train[0].extend(train[0])
            all_train[1].extend(train[1])
            all_test[0].extend(test[0])
            all_test[1].extend(test[1])

        self.__classifier.fit(all_train[0], all_train[1])

        predictions = self.__classifier.predict(all_test[0])
        predictions_train = self.__classifier.predict(all_train[0])

        print(metrics.classification_report_imbalanced(all_test[1], predictions))
        print(metrics.classification_report_imbalanced(all_train[1], predictions_train))

    def classify(self, filename: str, path: str):
        self.__reader.load_image(filename, path)

        new_img = np.zeros(self.__reader.get_img().shape)

        for i in [1, 2, 3, 4]:
            coords, x_args = self.__patch.get_quarter_patches(i)

            predictions = self.__classifier.predict(x_args)

            for j in range(len(coords)):
                new_img[coords[j][0], coords[j][1]] = predictions[j]

        return new_img