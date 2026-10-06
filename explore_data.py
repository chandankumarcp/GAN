"""Load and plot the CIFAR-10 training images."""
from tensorflow.keras.datasets.cifar10 import load_data
from matplotlib import pyplot


def main():
    (trainX, trainy), (testX, testy) = load_data()
    for i in range(49):
        pyplot.subplot(7, 7, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(trainX[i])
    pyplot.show()


if __name__ == '__main__':
    main()
