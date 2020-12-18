import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)


    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    """
    images = []
    labels = []
    print("Loading data...")
    for file in os.listdir(data_dir):
        path_to_file = os.path.join(data_dir,file)

        if os.path.isdir(path_to_file):
            for every_image in os.listdir(path_to_file):

                # read and resize images
                img = cv2.imread(os.path.join(path_to_file,every_image),cv2.IMREAD_COLOR)
                img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT),cv2.INTER_AREA)

                # images and labels as arrays
                images.append(img)
                labels.append(int(file))

        print(f'Category {file} loaded...')

    print("\n LOADING COMPLETE!!")

    return (images,labels)



def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.Sequential()

    # intial CONV layer
    model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    # multiple conv and pooling to reduce feature size
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(0.3))

    # final output = no of categories
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return (model)



if __name__ == "__main__":
    main()
