from ImgProcessor import ImgProcessor
from ImgReader import ImgReader

if __name__ == '__main__':
    reader = ImgReader()
    proc = ImgProcessor(reader)

    reader.load_image("01_dr", "../img/")
    proc.process()
    proc.show_img()
    proc.show_metrics()
