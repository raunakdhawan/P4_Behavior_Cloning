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

The Lambda layer at the start of the network is to normalize the input batch. So that the values are between -1 and 1. This is done by dividing by 255 and subtracting 1.0 from the result.

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

**Input image to the network** : First the top 65 pixels and bottom 130 pixels are removed to remove the horizon and the front of the car. The cropped image is the resized to an image of size (64, 64, 3). This is given as an input to the neural network.

### 4. Generator
A generator funciton is created in order to generate the batch of images to be fed into the neural network. It helps utilizing the memory efficiently. As the batches are being created here, the addition of recovery images is also done in the generator function.
1. First the data is shuffled (avoid biasing)
2. The images from the center, right and left cameras are read using the `cv2.imread`
3. The images are processed
    - Cropped
    - Resized
    - Converted from BGR to RGB (not really needed but helps in plotting)
4. The center image and the steering angle are added to the batch
5. Also, the center image is flipped horizontally and the steering angle is multiplied with -1, then added to the batch.
6. Recovery : The offset is added to the left camera image and added to the batch
7. Recovery : The offset is subtracted from the right camera image and added to the batch
8. Recovery : The left camera image is flipped horizontally and the calculated steering angle with the offset is multiplied with -1 and added to the batch
9. Recovery : The right camera image is flipped horizontally and the calculated steering angle with the offset is multiplied with -1 and added to the batch
10. The batch is yieldid.

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

The above model was trained only on the data provided by Udacity. After training and testing, it seemd that the data was not enough as the car was not recovering from the turns. Looking at the data from Udacity, the number of images with straight steering angles (== 0) were more than left or right steering angels. Hence the model was biased as the data is biases.

In order to remove the data bias, driving around the track was done and the images with the steering angles were recorded. Around three laps were done both the directions. This helped collect lots of unbiased data.

Also the model seemed to not detect changes in the road or the boundaries. With this model, the car was able to take the first left turn (as it can be done with smaller steering angles and the scene is not changing much), but once it reaches the bridge (even though it is straight) the car acts weird and the next left turn it does not detect the left, but sees the opening as the straight road. This meant creating a little bit more complicated NN. The neural network is adapted from the Nvidia neural network.

### 2. Creation of the Training Set & Training Process
- As mentioned before the training data was created by driving around the track atleast three times in both direction. This data was combined with the data provided by Udacity. 
- The data is divided into training set and validation set. The data is given as input to the neural network in a batch of 32 normalized images. 
- The input is provided to the neural network by using python generator so as to efficiently use the memory provided in the workspace. 

### 3. Final Solution
Here is the link to the [final resulting video](./output_video.mp4)