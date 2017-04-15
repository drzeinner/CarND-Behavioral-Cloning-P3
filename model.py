import argparse
import csv
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# -----------------------------------------------------
# Combine image directories and csv files
#
def combineData(src_dirs, dst):
    print("Combining data into {}".format(dst))

    dst_image_path = os.path.join(dst, 'IMG/')
    # First remove the contents in the dst_image_path
    if os.path.isdir(dst_image_path):
        shutil.rmtree(dst_image_path)
    # Now make the directory
    os.mkdir(dst_image_path)

    dst_csv_file_path = os.path.join(dst, 'driving_log.csv')
    # if the csv file exists, delete it
    if os.path.isfile(dst_csv_file_path):
        os.remove(dst_csv_file_path)

    # Now open the csv file to write
    with open(dst_csv_file_path, 'w') as dst_csv_file:
        writer = csv.writer(dst_csv_file)
        for subdir in src_dirs:
            # Copy the csv lines
            src_csv_file_path = os.path.join(subdir, 'driving_log.csv')
            if os.path.isfile(src_csv_file_path):
                with open(src_csv_file_path) as src_csv_file:
                    reader = csv.reader(src_csv_file)
                    for line in reader:
                        writer.writerow(line)
            # Copy the image folder
            subdir_path = os.path.join(subdir, 'IMG')
            if os.path.isdir(subdir_path):
                print("Copying {}".format(subdir_path))
                src_images = os.listdir(subdir_path)
                for image in src_images:
                    src_image_path = os.path.join(subdir_path, image)
                    shutil.copy2(src_image_path, dst_image_path)
        print("Data has been successfully copied into {}".format(dst))


# -----------------------------------------------------
# Augment Images
#
# Append flipped images and angles
#
def augmentData(images, measurements):

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)

    return augmented_images, augmented_measurements

# -----------------------------------------------------
# Apply some preprocessing to the image
# Grayscale
#
def preprocessImage(img):
    return img
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(160, 320, 1)

# Apply prepocessing to a list of images
#
def preprocessImages(imgs):
    preprocessed_imgs = []
    for img in imgs:
        preprocessed_imgs.append(preprocessImage(img))
    return preprocessed_imgs


# -----------------------------------------------------
# Create a generator to be able to load only the necessary images into memory
#
def generator(samples, data_directory, batch_size=32):
    num_samples = len(samples)
    angle_correction = 0.25
    image_directory = data_directory + 'IMG/'

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                steering_left = steering_center + angle_correction
                steering_right = steering_center - angle_correction

                center_source_path = batch_sample[0]
                center_filename = center_source_path.split('/')[-1]
                center_path = image_directory + center_filename
                img_center = np.asarray(cv2.imread(center_path))

                left_source_path = batch_sample[0]
                left_filename = left_source_path.split('/')[-1]
                left_path = image_directory + left_filename
                img_left = np.asarray(cv2.imread(left_path))

                right_source_path = batch_sample[0]
                right_filename = right_source_path.split('/')[-1]
                right_path = image_directory + right_filename
                img_right = np.asarray(cv2.imread(right_path))

                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

            # trim image to only see section with road
            augmented_images, augmented_angles = augmentData(images, angles)
            processImages = preprocessImages(augmented_images)
            X_train = np.array(processImages)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)


# -----------------------------------------------------
# Load data to use for training/validation
#
# param   dataDir   directory the data is located in
#
def loadData(dataDir):
    samples = []
    with open(os.path.join(dataDir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, dataDir, batch_size=32)
    validation_generator = generator(validation_samples, dataDir, batch_size=32)

    return train_samples, validation_samples, train_generator, validation_generator


# --------------------------------------------------------------------------
# Visualizes the training data
#
def visualizeTraining(history_object):

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# -----------------------------------------------
# train a neural network
#
def trainNetwork(train_data_dir, model_path):
    train_samples, validation_samples, train_generator, validation_generator = loadData(train_data_dir)
    print("num train_samples={} num validation_samples={}".format(len(train_samples), len(validation_samples)))

    # Convolutional 24 5x5 stride 2
    # Convolutional 36 5x5 stride 2
    # Convolutional 48 3x3 stride 1
    # Convolutional 64 3x3 stride 1
    # flatten
    # Dense 120
    # Dense 84
    # Dense 1

    model = Sequential()

    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
    model.save(model_path)
    visualizeTraining(history_object)

# ---------------------------------------------
# Execute Code Here
# Combine all of the data collected into a single directory
def main():
    parser = argparse.ArgumentParser(description="Train a neural network to drive a car")
    parser.add_argument('-l', '--list', help='delimited list input', type=lambda s: [item for item in s.split(' ')])
    parser.add_argument(
        'output_file',
        type=str,
        default='',
        help='Path to output model.'
    )
    args = parser.parse_args()

    combineData(args.list, 'trainData')
    trainNetwork('trainData/', args.output_file)

if __name__ == '__main__':
    main()