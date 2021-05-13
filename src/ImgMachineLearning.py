from ImgPatchExtractor import ImgPatchExtractor
from ImgReader import ImgReader
from sklearn import tree
from imblearn import metrics
import numpy as np
import os
import pickle


class ImgMachineLearning:
    def __init__(self, reader: ImgReader):
        self.__classifier = tree.DecisionTreeClassifier(random_state=0)
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

        for file in ["01_dr", "01_g", "01_h", "02_g"]:
            self.__reader.load_image(file, "../img/")

            self.__patch.extract_patches()

            train = self.__patch.get_train_sets()

            all_train[0].extend(train[0])
            all_train[1].extend(train[1])

        self.__classifier.fit(all_train[0], all_train[1])

    def classify(self, filename: str, path: str):
        self.__reader.load_image(filename, path)

        new_img = np.zeros(self.__reader.get_img().shape)

        for i in [1, 2, 3, 4]:
            coords, x_args = self.__patch.get_quarter_patches(i)

            predictions = self.__classifier.predict(x_args)

            for j in range(len(coords)):
                new_img[coords[j][0], coords[j][1]] = predictions[j]
            print("Zakończyłem ćwiartkę {}".format(i))
        return new_img

    def load_classifier(self, filename: str, path: str):
        with open(path + filename, 'rb') as file:
            self.__classifier = pickle.load(file)

    def dump_classifier(self, filename: str, path: str):
        with open(path + filename, 'wb') as file:
            pickle.dump(self.__classifier, file)
