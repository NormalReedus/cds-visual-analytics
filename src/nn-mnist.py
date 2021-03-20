import os
import sys
sys.path.append("..")
import argparse
from pathlib import Path

# Import teaching utils
import pandas as pd
import numpy as np
from utils.neuralnetwork import NeuralNetwork

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer



def load_mnist(data_path):
    img_path = os.path.join(data_path, 'mnist_img.csv')
    label_path = os.path.join(data_path, 'mnist_label.csv')

    if os.path.isfile(img_path) and os.path.isfile(label_path):
        img = pd.read_csv(img_path)
        label = pd.read_csv(label_path).squeeze() # Squeezes DataFrame into Series
    else:
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        img, label = fetch_openml('mnist_784', version=1, return_X_y=True)
        img.to_csv(img_path, sep=',', encoding='utf-8', index=False)
        label.to_csv(label_path, sep=',', encoding='utf-8', index=False)

    # We might need to excplicitly convert to numpy arrays for some versions of pandas and sklearn
    return (np.array(img), np.array(label))


def main(data_path, epochs):
    # Load data as np arrays
    img, label = load_mnist(data_path)

    # We are assuming the min and max values for pixel intensities
    # are between 0 and 255. The minmax normalization from session 7
    # might give values between say 10 and 230, which might not work
    # well when given a new image that has pixel values above or below those
    img = img / 255.0 # normalize pixel vals to between 0 and 1 as float


    classes = sorted(set(label))
    num_classes = len(classes)

    # Split our data 80/20 - train/test
    img_train, img_test, label_train, label_test = train_test_split(img, label, random_state=1337, test_size=0.2)

    # Convert labels to binary representation (e.g. 2 becomes [0,0,1,0,0,0,0,0,0,0])
    label_train = LabelBinarizer().fit_transform(label_train)
    label_test = LabelBinarizer().fit_transform(label_test)

    # Specify the neural network structure
    neural_network = NeuralNetwork([img_train.shape[1], 32, 16, num_classes]) # 1 input node for every pixel in images, 1 output node for every class

    # Train the model
    neural_network.fit(img_train, label_train, epochs=epochs)

    # Make predictions on all test images
    label_pred = neural_network.predict(img_test)
    label_pred = label_pred.argmax(axis=1) # Give us the highest probability label

    # Generate comparative metrics with test data
    classifier_metrics = metrics.classification_report(label_test.argmax(axis=1), label_pred)

    print(classifier_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train neural network on the full MNIST dataset and view the classifier metrics")

    parser.add_argument("-d", "--data_path", default = Path('../data/'), type = Path, help = "path to where the MNIST csv-files dataset is saved or where to save it")
    parser.add_argument("-e", "--epochs", default = 5, type = int, help = "numbers of epochs to train")

    args = parser.parse_args()
    
    main(data_path = args.data_path, epochs = args.epochs)