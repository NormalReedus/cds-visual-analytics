import os
import cv2
import sys
sys.path.append('..')
from utils.imutils import jimshow
import random
import matplotlib.pyplot as plt
import numpy as np

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def main():
    # Data directories
    train_dir = os.path.join('data', 'training')
    test_dir = os.path.join('data', 'validation')

    # Lists of all painters
    # these two lists are actually identical
    painters_train = os.listdir(train_dir)
    painters_test = os.listdir(test_dir)

    # Dictionary for training data
    # key: filename, value: label
    train_data = {}

    for painter in painters_train:
        # All training images for the painter
        files = os.listdir(os.path.join(train_dir, painter))

        # Add path_to_image: painter_name to dictionary
        for file in files:
            filepath = os.path.join(train_dir, painter, file)
            train_data[filepath] = painter

    # Dictionary for validation data
    # key: filename, value: label
    test_data = {}

    for painter in painters_test:
        # All training images for the painter
        files = os.listdir(os.path.join(test_dir, painter))
        
        # Add path_to_image: painter_name to dictionary
        for file in files:
            filepath = os.path.join(test_dir, painter, file)
            test_data[filepath] = painter

    # Convert to list structure that tensorflow wants
    train_img, train_label = zip(*train_data.items())
    test_img, test_label = zip(*test_data.items())

    # Generate random indices for samples
    random_train_indices = random.sample(range(0, len(train_img)), 1000)
    random_test_indices = random.sample(range(0, len(test_img)), 320)

    # New lists of samples from the indices
    train_img = [file for index, file in enumerate(train_img) if index in random_train_indices]
    train_label = [label for index, label in enumerate(train_label) if index in random_train_indices]
    test_img = [file for index, file in enumerate(test_img) if index in random_test_indices]
    test_label = [label for index, label in enumerate(test_label) if index in random_test_indices]

    # Load images from filepaths
    train_img = [cv2.imread(filepath) for filepath in train_img]
    test_img = [cv2.imread(filepath) for filepath in test_img]

    # Resize all images to same size
    target_size = [150, 150]
    train_img = [cv2.resize(img, tuple(target_size), interpolation = cv2.INTER_AREA) for img in train_img]
    test_img = [cv2.resize(img, tuple(target_size), interpolation = cv2.INTER_AREA) for img in test_img]

    # Normalize pixel data
    train_img = np.array(train_img).astype('float')
    test_img = np.array(test_img).astype('float')
    train_img /= 255.0
    test_img /= 255.0

    # Binarize labels
    lb = LabelBinarizer()
    train_label = lb.fit_transform(train_label)
    test_label = lb.fit_transform(test_label)

    # List of labels - LabelBinarizer's generated labels are sorted alphabetically
    labels_alphabetized = sorted(painters_train)

    # Dimensions of image in the model + 3 color channels 
    image_shape = target_size
    image_shape.append(3)
    image_shape = tuple(image_shape)

    # Define LeNet model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), 
                    padding="same", 
                    input_shape=image_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), 
                    padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                optimizer=SGD(lr=0.01),
                metrics=["accuracy"])

    # Take a look at the architecture
    model.summary()

    # Train the model
    trained_model = model.fit(train_img, train_label, 
                validation_data=(test_img, test_label), 
                batch_size=32,
                epochs=20,
                verbose=1)

    # Save report of metrics
    predictions = model.predict(test_img, batch_size=32)
    report = classification_report(test_label.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels_alphabetized)

    with open(os.path.join('output', 'report.txt'), 'w') as file:
        file.write(report)

    # Save history plot
    def plot_history(H, epochs):
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('output', 'loss_and_accuracy.png'))
        plt.show()

    plot_history(trained_model, 20)

if __name__ == '__main__':
    main()