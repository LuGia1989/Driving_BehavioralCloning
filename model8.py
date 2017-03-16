import csv
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg


def augment_brightness(image):
    """
    This function is to adjust the brightness of the image.

    First, I convert the image to HSV format.
    Using the numpy random uniform function to generate the random number between 0 and 1.
    Use that as the multiplication factor to adjust the brightness Value of HSV image.

    When that's done, I convert back to RGB format

    :param image:
    :return: image
    """
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform()
    image[:, :, 2] = image[:, :, 2] *random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image



def plot_images ():
    """
    This function is to display six sample images.

    The 5th sample image also displays with the brightness adjusted.
    :return:
    """
    sample_dir = 'sample_images/'
    for i in range(6):
        image = mpimg.imread(sample_dir + 'image_%d.jpg' % (i + 1))
        if i == 5:
            image = augment_brightness(image)
        plt.rcParams["figure.figsize"] = [12, 4]
        plt.subplot(2,3,i+1)
        plt.imshow(image, aspect='auto')
        plt.axis('off')
    plt.suptitle('Explore Some Image Data')
    plt.show()


def data_exploration(samples, text = ''):
    """
    This function is to plot the Steering distrubution across samples

    :param samples:
    :param text:
    :return:
    """
    steerings = []
    for steer in samples:
        steering = float(steer[3])
        steerings.append(steering)

    # Plot the histogram
    plt.rcParams["figure.figsize"] = [10, 5]
    axes = plt.gca()
    plt.hist(steerings, bins=20)
    plt.title('Steering Distribution across ' + text)
    plt.show()

def data_extraction(filename):
    """
    This function reads csv file and append to the list.  Return the sample list
    :param filename:
    :return: samples list
    """
    samples = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def data_flip(image, angle):
    """
    This function flipped center data - horizontal flip
    :param image:
    :param angle:
    :return: flipped image and flipped angle
    """
    flipped_image = np.fliplr(image)
    flipped_angle = -angle
    return flipped_image, flipped_angle



def generator(samples, train_mode = 0, validation_mode = 0, batch_size=32):
    """
    I use all three cameras center, left, right for train data.  For the left camera, I ADD angle coeff.
    For the right camera, I SUBTRACT with angle coeff.
    :param samples:
    :param train_mode:
    :param validation_mode:
    :param batch_size:
    :return: X data and y labels
    """
    angle_coeff = 0.12
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if train_mode == 1:
                    # train data use augmented data: 3 augmented data for center/left/right with angle coef, and 1
                    # flipped center data
                    left = './data/IMG/' + batch_sample[1].split('/')[-1]
                    center = './data/IMG/'+ batch_sample[0].split('/')[-1]
                    right = './data/IMG/' + batch_sample[2].split('/')[-1]

                    left_image = cv2.imread(left)
                    center_image = cv2.imread(center)
                    right_image = cv2.imread(right)
                    augmented_brightness_center_image = augment_brightness(center_image)


                    left_angle = float(batch_sample[3]) + angle_coeff
                    center_angle = float(batch_sample[3])
                    augmented_brightness_center_angle = float(batch_sample[3])
                    right_angle = float(batch_sample[3]) - angle_coeff

                    # flipped center data - horizontal flip
                    flipped_center_image, flipped_center_angle = data_flip(center_image,center_angle)

                    images.extend([left_image, center_image, right_image, flipped_center_image, augmented_brightness_center_image])
                    angles.extend([left_angle, center_angle, right_angle, flipped_center_angle, augmented_brightness_center_angle])

                elif validation_mode == 1:
                    # validation data don't need to use augmented data
                    center = './data/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center)
                    center_angle = float(batch_sample[3])
                    images.extend([center_image])
                    angles.extend([center_angle])

            # convert the data to numpy array prior to feed into Keras
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def my_model():
    """
    This function model the NVDIA architecture as described in
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    :return: architecture model
    """
    ch, row, col = 3, 160, 320
    #learning_rate =  0.0001
    #dropout_prob = 0.5

    model = Sequential()
    model.add(Cropping2D(cropping=((45, 20), (0, 0)), input_shape=(row, col, ch)))
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1.))
    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    #model.add(Dropout(dropout_prob))
    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


def train_model(model, train_samples, validation_samples, train_generator, validation_generator):
    """
    samples_per_epoch is 5 times the train samples because for each train data, i augmented to 3 more augmented data:
    center/left/right with angle coefficence, flipped center data, and augmented brightness data
    :param model:
    :param train_samples:
    :param validation_samples:
    :param train_generator:
    :param validation_generator:
    :return: history_object
    """
    history_object =model.fit_generator(train_generator, samples_per_epoch= len(5*train_samples), \
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), \
                    nb_epoch=5, verbose=1)

    model.save("model8.h5")
    return history_object

def plot (history_object):
    """
    This function plots the train loss and validation loss
    :param history_object:
    :return:
    """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss - model 8')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    activation_relu = 'relu'
    filename = './data/driving_log.csv'

    # Data extraction
    samples = data_extraction(filename)

    # Data exploration
    data_exploration(samples, text='entire samples')
    plot_images()

    # Split the train and validation to 80/20 ratio
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    data_exploration(train_samples, text='train samples')
    data_exploration(validation_samples, text = 'validation samples')


    # create train and validation data from gen
    train_generator = generator(train_samples, train_mode = 1, validation_mode = 0, batch_size=32)

    validation_generator = generator(validation_samples, train_mode = 0, validation_mode = 1, batch_size=32)

    # Call the NVDIA model
    model = my_model()

    # Train the model
    train_start_time = time.time()
    history_object = train_model(model, train_samples, validation_samples, train_generator, validation_generator)
    total_time = time.time() - train_start_time
    print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time / 60))

    # Plot the train vs. validation loss
    plot(history_object)

