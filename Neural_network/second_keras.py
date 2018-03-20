import tensorflow as tf       # tenserflow
from matplotlib import pyplot as plt
import numpy as np
from mnist import MNIST
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cv2

def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
          cv2.circle(img,(x,y),20,(255,255,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False

def plot_preds(image, preds):
  plt.figure(1,figsize=(13, 8))
  plt.subplot(121)
  plt.imshow(image)
  plt.axis('on')
  order = np.arange(len(preds))
  bar_preds = preds
  labels = [x for x in range(0,10)]
  x = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
  [labels.append(x[i]) for i in range(0,26)]
  plt.subplot(122)
  plt.barh(order, bar_preds, 0.3)
  plt.yticks(order, labels)
  plt.xlabel('Probability [%]')
  plt.xlim(0,100)
  plt.show()
  return bar_preds,preds

def result(tryb,numer,img):
    test = np.empty((1,28,28))
    if tryb==1:
      test[0]=testimagesMNIST[numer]
    else:
      test[0]=img
    test = test.reshape(test.shape[0],1,28,28)
    test = test.astype('float32')
    test /= 255
    if tryb==1:
      print('Number to guess: {}'.format(testlabelsMNIST[numer]))
    wynik = model.predict_on_batch(test)
    result = np.empty((36))
    for i in range(0,36):
        result[i] = round(wynik[0][i],4)
        result[i] = result[i]*100
        if i<10:
            print('{}: {} %'.format(i, result[i]))
        else:
            x = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
            print('{}: {} %'.format(x[i-10], result[i]))
    print('\n')
    if tryb==1:
      plot_preds(testimagesMNIST[numer],result)
    else:
      plot_preds(img,result)
    return result

drawing=False # true if mouse is pressed
img = np.zeros((560,560), np.uint8)
cv2.namedWindow('Window')
cv2.setMouseCallback('Window',interactive_drawing)

model = load_model('C:/Users/Mateusz/Dropbox/Python/Neural_network/digits_new_128.h5')

###C:/Users/Mateusz/Dropbox/Python/Neural_network
##mndata = MNIST('C:/Users/Mateusz/Dropbox/Python/Neural_network/Data/')
##imagess, labelss = mndata.load_training()
##testimagesMNIST = np.empty((10000,28,28))
##testlabelsMNIST = np.empty((10000))
##
##for x in range(49999,59999):
##    image=np.array(imagess[x],dtype=np.uint8)
##    image.resize(28,28)
##    testimagesMNIST[x-49999]=image
##    testlabelsMNIST[x-49999]=labelss[x]
##
##emndata = MNIST('C:/Users/Mateusz/Dropbox/Python/Neural_network/emnist_data/')
##emndata.select_emnist('letters')
##emndata.gz = False
##images, labels = emndata.load_training()
##testimagesEMNIST = np.empty((24700,28,28))
##testlabelsEMNIST = np.empty((100000))
##
##for x in range(99999,123999):
##    image=np.array(images[x],dtype=np.uint8)
##    image.resize(28,28)
##    testimagesEMNIST[x-99999]=image
##    testlabelsEMNIST[x-99999]=labels[x]
##
##
##D_test = testimagesMNIST.reshape(testimagesMNIST.shape[0],1,28,28)
##L_test = testimagesEMNIST.reshape(testimagesEMNIST.shape[0],1,28,28)
##
### convert to float32 and normalize [0,1]
##D_test = D_test.astype('float32')
##D_test /= 255
##L_test = L_test.astype('float32')
##L_test /= 255
##
###one line hot encoding 
##d_test = np_utils.to_categorical(testlabelsMNIST,10)
##l_test = np_utils.to_categorical(testlabelsMNIST,10)


#score = model.evaluate(D_test, d_test, batch_size=1000)

def get_result():
  while True:
      cv2.imshow('Window',img)
      if cv2.waitKey(33) == ord('a'):
          break
  cv2.destroyAllWindows()

while True:
  get_result()
  img = cv2.resize(img,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
  result(2,2,img)
  img = np.zeros((560,560), np.uint8)
  cv2.namedWindow('Window')
  cv2.setMouseCallback('Window',interactive_drawing)

    








