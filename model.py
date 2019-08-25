import csv
import cv2
import numpy as np
import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda


# Read the data log from csv file
lines = []
data_log_csv = './data/driving_log.csv'
with open(data_log_csv) as data_log:
    read_lines = csv.reader(data_log)
    for line in read_lines:
        lines.append(line)

# Crop and Resize the images
def pre_process_image(image):
    return cv2.cvtColor(cv2.resize(image[80:140,:], (32,32)), cv2.COLOR_BGR2RGB)

# Generator for the images
def generator_images(data, batchSize = 32, steering_offset=0.2):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), int(batchSize/4)):
            X_batch = []
            y_batch = []
            details = data[i: i+int(batchSize/4)]
            for line in details:
                image = pre_process_image(cv2.imread('./data/IMG/'+ line[0].split('/')[-1]))
                steering_angle = float(line[3])
                #appending original image
                X_batch.append(image)
                y_batch.append(steering_angle)
                #appending flipped image
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)
                # appending left camera image and steering angle with offset
                X_batch.append(pre_process_image(cv2.imread('./data/IMG/'+ line[1].split('/')[-1])))
                y_batch.append(steering_angle + steering_offset)
                # appending right camera image and steering angle with offset
                X_batch.append(pre_process_image(cv2.imread('./data/IMG/'+ line[2].split('/')[-1])))
                y_batch.append(steering_angle - steering_offset)
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)

# Create Training and Validation sets
training_data, validatio_data = train_test_split(lines, test_size = 0.2)

# Create the model
model = Sequential()
model.add(Lambda(lambda x: x /255.0, input_shape=(32,32,3) ))  # With data normalization
model.add(Convolution2D(15, 3, 3, subsample=(2, 2), activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1))

# Compile and fit the model
model.compile('adam', 'mse')
model.fit_generator(generator_images(training_data), 
                    samples_per_epoch = len(training_data)*4, 
                    nb_epoch = 2, 
                    validation_data=generator_images(validatio_data), 
                    nb_val_samples=len(validatio_data))

#saving the model
model.save('model.h5')