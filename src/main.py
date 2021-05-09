from ImgProcessor import ImgProcessor
from ImgReader import ImgReader
from ImgPatchExtractor import ImgPatchExtractor
from ImgMachineLearning import ImgMachineLearning
import skimage.io as io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    reader = ImgReader()
    proc = ImgProcessor(reader)

    reader.load_image("02_dr", "../img/")
    # proc.process()
    # proc.show_img()
    # proc.show_metrics()

    # patch = ImgPatchExtractor(reader)
    # patch.extract_patches()
    #
    # train, test = patch.get_train_test_sets()
    #
    # print(train, test)

    machine = ImgMachineLearning(reader)

    machine.train()

    img = machine.classify("02_dr", "../img/")
    io.imshow(img)
    plt.show()
