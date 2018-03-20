import tensorflow as tf       # tenserflow
from matplotlib import pyplot as plt
import numpy as np
from mnist import MNIST
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets # mnist library (digits)
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Download images and labels

#MNIST
mndata = MNIST('C:/Users/Mateusz/Dropbox/Python/Neural_network/Data/')
images, labels = mndata.load_training()
imagesMNIST = np.empty((59000,28,28))
labelsMNIST = np.empty((59000))
testimagesMNIST = np.empty((1000,28,28))
testlabelsMNIST = np.empty((1000))
for x in range(0,58999):
    image=np.array(images[x],dtype=np.uint8)
    image.resize(28,28)
    imagesMNIST[x]=image
    labelsMNIST[x]=labels[x]

for x in range(58999,59999):
    image=np.array(images[x],dtype=np.uint8)
    image.resize(28,28)
    testimagesMNIST[x-58999]=image
    testlabelsMNIST[x-58999]=labels[x]

#EMNIST
emndata = MNIST('C:/Users/Mateusz/Dropbox/Python/Neural_network/emnist_data/')
#C:\Users\Mateusz\Dropbox\Python\Neural_network
#D:/Dropbox/Python/Neural_network/emnist_data/
emndata.select_emnist('letters')
emndata.gz = False
images, labels = emndata.load_training()
imagesEMNIST = np.empty((124000,28,28))
labelsEMNIST = np.empty((124000))
testimagesEMNIST = np.empty((700,28,28))
testlabelsEMNIST = np.empty((700))

for x in range(0,123999):
    image=np.array(images[x],dtype=np.uint8)
    image.resize(28,28)
    imagesEMNIST[x]=image
    labelsEMNIST[x]=labels[x]

for x in range(123999,124699):
    image=np.array(images[x],dtype=np.uint8)
    image.resize(28,28)
    testimagesEMNIST[x-123999]=image
    testlabelsEMNIST[x-123999]=labels[x]

D_train = imagesMNIST.reshape(imagesMNIST.shape[0],1,28,28)
L_train = imagesEMNIST.reshape(imagesEMNIST.shape[0],1,28,28)
D_test = testimagesMNIST.reshape(testimagesMNIST.shape[0],1,28,28)
L_test = testimagesEMNIST.reshape(testimagesEMNIST.shape[0],1,28,28)

# convert to float32 and normalize [0,1]
D_train = D_train.astype('float32')
D_test = D_test.astype('float32')
D_train /= 255
D_test /= 255
L_train = L_train.astype('float32')
L_test = L_test.astype('float32')
L_train /= 255
L_test /= 255

labelsEMNIST=labelsEMNIST+9 # przesuwamy one line hot encoding o 10 w prawo
testlabelsEMNIST=testlabelsEMNIST+9 
train = np.append(labelsMNIST,labelsEMNIST,axis=0) 
test = np.append(testlabelsMNIST,testlabelsEMNIST,axis=0) 

Train = np.append(D_train,L_train,axis=0)           # train = zdjecia
Test = np.append(D_test,L_test,axis=0)              # train = zdjecia testowe

#one line hot encoding
train = np_utils.to_categorical(train,36)           # test = etykiety testowe
test = np_utils.to_categorical(test,36)             # train = etykiety
   
#defining model architecture
model = Sequential()
model.add(Convolution2D(32, (2,2), activation='relu', input_shape=(1,28,28),data_format='channels_first')) #depth, width, height

model.add(Convolution2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())                        # weights from the Convolution layers must be flattened (made 1-dimensional) 
model.add(Dense(128, activation='relu'))    # output size of file
model.add(Dropout(0.25))
model.add(Dense(36, activation='softmax'))  # 36 = 36 digits 

# loss function
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(Train, train,batch_size=64, epochs=30, verbose=1)

model.save('C:/Users/Mateusz/Dropbox/Python/Neural_network/digits_new_64relu.h5')
