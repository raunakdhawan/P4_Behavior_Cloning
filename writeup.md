# **Behavioral Cloning** 

## Model Architecture

### 1. Architecture
The following is the architecture summary that is implemented for this project

| Layer (type)        | Output Shape       | Param # |
| :------------------ | :----------------- | :------ |
| lambda_1 (Lambda)   | (None, 64, 64, 3)  | 0       |
| conv2d_1 (Conv2D)   | (None, 30, 30, 24) | 1824    |
| conv2d_2 (Conv2D)   | (None, 13, 13, 36) | 21636   |
| conv2d_3 (Conv2D)   | (None, 5, 5, 48)   | 43248   |
| conv2d_4 (Conv2D)   | (None, 3, 3, 64)   | 27712   |
| conv2d_5 (Conv2D)   | (None, 1, 1, 64)   | 36928   |
| flatten_1 (Flatten) | (None, 64)         | 0       |
| dense_1 (Dense)     | (None, 80)         | 5200    |
| dropout_1 (Dropout) | (None, 80)         | 0       |
| dense_2 (Dense)     | (None, 40)         | 3240    |
| dropout_2 (Dropout) | (None, 40)         | 0       |
| dense_3 (Dense)     | (None, 16)         | 656     |
| dropout_3 (Dropout) | (None, 16)         | 0       |
| dense_4 (Dense)     | (None, 10)         | 170     |
| dense_5 (Dense)     | (None, 1)          | 11      |

Total params: 140,625
Trainable params: 140,625
Non-trainable params: 0

### 2. Avoiding of Overfitting
In order to avoid overfitting, dropouts are introduced in the Fully Connected Layes. Also in addition to dropout, L2 regularization is introduced, as the network will be trained on 9 laps of the same track with 10 epochs. With both the Dropout and the L2 Regularization, the Overfitting is reduced.

### 3. Model Parameter
The following parameters are used in the model
| Hyper Parameter    | Value              |
| :----------------- | :----------------- |
| Batch Size         | 32                 |
| Epochs             | 10                 |
| Optimizer          | Adam               |
| Learning Rate      | 0.0001             |
| Steering Offset    | 0.25               |
| Resized Image Size | 64, 64, 3          |
| Dropout            | 0.4                |
| L2 Regularization  | 0.001              |
| Validation Size    | 0.2                |
| Loss Function      | Meas Squared Error |

## Model Architecture and Training Strategy

### 1. Solution Design Approach
The first step was to just create a base model and get a baseline that I would use to measure the improvement of my changes in the model and training strategy. The base model created had the following architecture
| Layer (type)                  | Output Shape       | Param # |
| :---------------------------- | :----------------- | :------ |
| lambda_1 (Lambda)             | (None, 64, 64, 3)  | 0       |
| conv2d_1 (Conv2D)             | (None, 31, 31, 15) | 420     |
| conv2d_2 (Conv2D)             | (None, 14, 14, 24) | 9024    |
| dropout_1 (Dropout)           | (None, 14, 14, 24) | 0       |
| max_pooling2d_1 (MaxPooling2) | (None, 7, 7, 24)   | 0       |
| flatten_1 (Flatten)           | (None, 1176)       | 0       |
| dense_1 (Dense)               | (None, 10)         | 170     |
| dense_1 (Dense)               | (None, 1)          | 11      |

The above model was trained only on the data provided by Udacity

### 2. Creation of the Training Set & Training Process

### 3. Final Solution