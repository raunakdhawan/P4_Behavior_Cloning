## Import the required modules
import csv
import cv2
import numpy as np
import datetime
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda, ELU, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import math


# Read the csv log file
def read_csv_log(csv_path):
    # Read the data log from csv file
    lines = []
    with open(csv_path) as data_log:
        read_lines = csv.reader(data_log)
        for line in read_lines:
            if 'center' in line:
                continue
            else:
                lines.append(line)
    return lines            

# Crop and Resize the images
def pre_process_image(image, top, bottom, row, col):
    image_cropped = image[top:bottom,:]
    return cv2.cvtColor(cv2.resize(image_cropped, (col, row)), cv2.COLOR_BGR2RGB)

# Generator to yield images in batches
# def generator_images(data, img_path, row=32, col=32, batchSize=32, steering_offset=0.25):
def generator_images(data, img_path, top=65, bottom=130, row=64, col=64, batchSize=32, steering_offset=0.25):
    while True:
        data = shuffle(data)
        for i in range(0, len(data), batchSize):
            X_batch = []
            y_batch = []
            for line in data[i:i+batchSize]:
                image_center_path = img_path + line[0].split('/')[-1]
                image_left_path = img_path + line[1].split('/')[-1]
                image_right_path = img_path + line[2].split('/')[-1]
              
                if not os.path.isfile(image_center_path) \
                    or not os.path.isfile(image_left_path) \
                    or not os.path.isfile(image_right_path):
                    print("File Not Found!\n")
                    continue
                
                # Center Camera
                image = cv2.imread(image_center_path)
                image = pre_process_image(image, top, bottom, row, col)
                steering_angle = float(line[3])
                X_batch.append(image)
                y_batch.append(steering_angle)
                
                # Center Camera with flipped Image
                X_batch.append(np.fliplr(image))
                y_batch.append(-steering_angle)
                
                # Left Camera
                image_l = cv2.imread(image_left_path)
                image_l = pre_process_image(image_l, top, bottom, row, col)
                X_batch.append(image_l)
                y_batch.append(steering_angle + steering_offset)
                    

                # Left Camera with flipped Image
                X_batch.append(np.fliplr(image_l))
                y_batch.append(-steering_angle - steering_offset)
                
                # Right Camera
                image_r = cv2.imread(image_right_path)
                image_r = pre_process_image(image_r, top, bottom, row, col)
                X_batch.append(image_r)
                y_batch.append(steering_angle - steering_offset)
                

                # Right Camera with flipped Image
                X_batch.append(np.fliplr(image_r))
                y_batch.append(-steering_angle + steering_offset)
            
            # converting to numpy array
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield shuffle(X_batch, y_batch)


def base_model(row, col, dropout_rate, top=65, bottom=130, image_shape=(160, 320, 3)):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, 3)))  # Normalize Data
    model.add(Cropping2D(cropping=((top, bottom), (0, 0)), input_shape=image_shape)) # crop off top and bottom pixels
    model.add(Conv2D(15, (3, 3), 
                     activation="relu", 
                     strides=(2, 2)))
    model.add(Conv2D(24, (5, 5), 
                     activation='relu',
                     strides=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(80, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(40,activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dense(1))
    return model

def model_nvidia(row, col, dropout_rate,):
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, 3)))
    model.add(Conv2D(24, (5, 5), 
                     kernel_regularizer = l2(0.001), 
                     strides=(2, 2), 
                     padding="valid",
                     activation='relu'))
    model.add(Conv2D(36, (5, 5), 
                     kernel_regularizer = l2(0.001), 
                     strides=(2, 2), 
                     padding="valid", 
                     activation='relu'))
    model.add(Conv2D(48, (5, 5), 
                     kernel_regularizer = l2(0.001), 
                     strides=(2, 2), 
                     padding="valid", 
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), 
                     kernel_regularizer = l2(0.001), 
                     strides=(2, 2), 
                     padding="same", 
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), 
                     kernel_regularizer = l2(0.001), 
                     strides=(2, 2), 
                     padding="valid", 
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(80, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(40,activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu', kernel_regularizer = l2(0.001)))
    model.add(Dense(1, kernel_regularizer = l2(0.001)))
    
    return model


if __name__ == "__main__":
    # Path to the csv log
    data_log_csv = './data/driving_log.csv'
    # data_log_csv = './data/driving_log.csv'

    # Path to the images folder
    data_img = './data/IMG/'
    # data_img = './data/IMG/'
    
    # Hyper Parameters for the training
    BATCH_SIZE = 32
    ALPHA = 0.0001
    EPOCHS = 10
    DROPOUT = 0.4
    STEER_OFFSET = 0.3
    top = 65
    bottom = 130
    row = 64
    col = 64
#     image_shape = (160, 320, 3)
    
    # Read the csv data
    lines = read_csv_log(data_log_csv)
    
    # Create Training and Validation sets
    training_data, validation_data = train_test_split(lines, test_size = 0.2)

    # Create the model
#     model = base_model(row, col, DROPOUT)
    model = model_nvidia(row, col, DROPOUT)
    
    # Compile the model and fit using the Generator
    adam = Adam(lr=ALPHA)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
#     model.fit_generator(generator_images(training_data, data_img, row, col, BATCH_SIZE, STEER_OFFSET),
    model.fit_generator(generator_images(training_data, data_img, top, bottom, row, col, BATCH_SIZE, STEER_OFFSET),
                        steps_per_epoch=math.ceil(len(training_data)/BATCH_SIZE), 
                        epochs=EPOCHS, 
                        validation_data=generator_images(validation_data, data_img, top, bottom, row, col, BATCH_SIZE, STEER_OFFSET), 
                        validation_steps=len(validation_data))

    # Save the model
#     model.save('model.h5')
#     model.save('model_nvidia.h5')
    model.save('test.h5')