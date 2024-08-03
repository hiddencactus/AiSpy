import tensorflow as tf;
from tensorflow import keras
import os                   #os is used to navigate through file structure      os.listdir('data) will list everything in the data directory
import cv2                  #opencv computer vision stuff
import imghdr               #determines the type of image
from matplotlib import pyplot as plt
import numpy as np
from keras import Sequential, layers, metrics       # tensorflow.python.keras is deprecated
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout    #Conv2D is the convolutional neural network, MaxPoolings acts as condensing layer (whats max value in this region? return just that max value) 
                                                                                            #Dense is a fully connected layer (performs linear transformations on input data)
                                                                                            #Flatten removes all the dimensions exact one for inputting into next layer
                                                                                            #Dropout randomly sets input uits to 0 to prevent overfitting
    #THIS SHIT DOESNT FUCKING WORK ANYMORE FUCK YOU KERAS RATS --you need to use layers.Conv2D directly now


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(len(gpus)) #no nvidia gpu unfortunately....

data_dir = 'aithingy\\data'                               #directory name
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#print(os.listdir(os.path.join(data_dir, 'happy')))                             #copy relative path. join method will concate the provided paths
                                                                               #anything under 10 kb, just delete the file, its probably a corrupted or improper image


#---------------------------------------------TEST
# print(os.path.exists(os.path.join(data_dir,'happy','image14.jpeg')))            #***need to add full file extension
# print(cv2.imread(os.path.join(data_dir,'happy','image14.jpeg')))
# img = cv2.imread(os.path.join(data_dir,'happy','image14.jpeg'))
# plt.imshow(img)                                                                # imshow displays the image  (imread/imshow)
# plt.show()                                                                     # displays the image
#----------------------------------------------

#delete dodgy files
for image_class in os.listdir(data_dir):                    #image class is /happy or /sad
    for image in os.listdir(os.path.join(data_dir, image_class)):   #image is an image like sadness.jpg
        #print(image)
        image_path = os.path.join(data_dir, image_class, image)     #image path
        try:
            img = cv2.imread(image_path)                            #read the image using opencv, produces numpy array 
            tip = imghdr.what(image_path)                           #determines the image type
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)                               #deletes the file
        except Exception as e:
            print('Issue with image {}'.format(image_path))         

#load data into dataset (using tensorflow dataset api)
data = tf.keras.utils.image_dataset_from_directory(data_dir)        #reads images from the directory, labels the images, decodes them into tensors and resizes them to a common shape, creates a dataset
#print(data)

# data_iterator = data.as_numpy_iterator()        #creates an iterator
# batch = data_iterator.next()                    #grabbing a batch back from data pipeline
#print(batch[1])                                 #len(batch) returns 2. A batch consists of samples, which are 1. the images loaded from the directories and 2. the labels
                                                #prints [1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 0 1 1], 1 and 0's assigned in alphabetical order
                                                #hence, happy is assigned 0, sad assigned 1
class_names = data.class_names
print(class_names)

#-------------------------------------------------------- 4 IMAGES FROM BATCH VISUALIZATION CLASS 0 = HAPPY, CLASS 1 = SAD
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):        #nested loop combo, captures tuples [['alice', 1], ['bob', 2]]  ----batch[0] refers to images, batch[0][:4] gets first 4 images
#     ax[idx].imshow(img)             #displays image on subplot
#     ax[idx].title.set_text(batch[1][idx])       #text set to the label
#plt.show()                                     #make sure you add this to show the plot
#--------------------------------------------------------

#PART 2: preprocessing the images

#basically, certain images can suck and they might be too dark or too might (under or over exposure)... this is basically like making the contrast very even in the image.
#its called histogram equalization, we normalize the image from 0-255 possible lightness values to 0-1
#something about the values being too high if its 255... i guess keras is too weak to handle it? General idea is smaller numbers = model go faster

# scaled = batch[0] / 255
# print(scaled.max())                       #instead of applying this afterward, apply the transformation directly to the data

#scale data
data = data.map(lambda x,y: (x/255, y))     #map will apply the anonymous function to each batch (an set of images and set of labels) for each set of images and labels, apply the transformation
data_iterator = data.as_numpy_iterator()        #creates an iterator
batch = data_iterator.next()                    #grabbing a batch back from data pipeline
#print(batch)                                    #CAN MOVE THIS TO BEFORE TO MAKE SIMPLER

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):        #nested loop combo, captures tuples [['alice', 1], ['bob', 2]]  ----batch[0] refers to images, batch[0][:4] gets first 4 images
#     ax[idx].imshow(img)             #displays image on subplot
#     ax[idx].title.set_text(batch[1][idx])       #text set to the label
# plt.show()        

print(len(data))    #how many batches are available

#partitioning training set
train_size = int(len(data)*.7) #setting size of the training model, validation model, and test model (70% of of the data)
val_size = int(len(data)*.2)+1  #training data is going to be used to train the model, validation is used to validate the model (fine tune the model), test partition is used post training
test_size = int(len(data)*.1)+1
print(train_size, val_size, test_size) #should add up to the size of data object
                                       #2 * 32 + 1 * 32 + 1 * 32

train = data.take(train_size)           #take defines how many batches we are going to take for that partition
val = data.skip(train_size).take(val_size)  #skip the batches that we already allocated to the training partition, and allocate the next batches
test = data.skip(train_size+val_size).take(test_size)

#---------------------------------------------------------------    BUILDING THE CONVOLUTIONAL MODEL
model = Sequential()

#                convolution has 16 filters, 3 pixels by 3 pixels, a stride of 1 (moves forward by 1 pixel) ---you can changes these, these are architectural decisions
#                relu activation just takes all of the output from a layer and passes it through a relu function
#                the keras.utils.image_dataset_from_directory() function reshapes the images to this size
model.add(layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))         #adds a convolutional layer   (model.add adds layers) --- the first layer needs an input
model.add(layers.MaxPooling2D())                                                           #adding max pooling layer   (scans across the relu activation and returns a number)
                                                                                    #*** maxpooling halves the input because it looks at a 2x2 block of input, and its stride is also 2x2
model.add(layers.Conv2D(32, (3,3), 1, activation='relu'))  #32 filters, 3 by 3 pixels, stride of 1
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(16, (3,3), 1, activation='relu')) 
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())                                #flattening the 3 layers down (from input shape you can see there are 3 channel values (RGB), flatten into 1)

model.add(layers.Dense(256, activation='relu'))            #fully connected layers (condensing the output)
model.add(layers.Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])    #adam is the optimizer we want, loss is BinaryCrossentropy since its a binary classification problem
model.summary()                                                                     #accuracy will tell us how well we are classifying as 0 or 1

# Model: "sequential"                        <-----------------------this should be the output
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 254, 254, 16)      448
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 127, 127, 16)      0              <------------- 0 means non trainable layer, just condenses things down
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 125, 125, 32)      4640
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 62, 62, 32)        0              
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 60, 60, 16)        4624
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 30, 30, 16)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 14400)             0               <------------- 14400 = 30 * 30 * 16  (converting multirank tensor into 1 dimension)
# _________________________________________________________________
# dense (Dense)                (None, 256)               3686656
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 257
# =================================================================
# Total params: 3,696,625
# Trainable params: 3,696,625
# Non-trainable params: 0


#------ TRAINING THE MODEL
logdir='aithingy\logs'           #path to log directory
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)   #saves model at checkpoint, callback is not a callback function, in tensorflow its an object that can perform specific actions at various stages of training
                                                                        # at start or end of an epoch, before or after a single batch is processed, when training begins or ends
                                                                        #the checkpoint is important, because if you end up overtraining / overfitting and the accuracy worsens, you can return to previous checkpoint

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])   #fit is the training method, predict is the prediction method
                # train is the training data, epoch is how long we train for (1 epoch is 1 run over the dataset) 
                #val is the validation data we set up before
                #pass through callback


#---------------------- EVALUATING PERFORMANCE
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')      #takes an array and plots it on a graph
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')    #loss quantifies the difference between the predicted values and true values. Accuracy is proportion of correct guesses
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')      #takes an array and plots it on a graph
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')    #loss quantifies the difference between the predicted values and true values. Accuracy is proportion of correct guesses
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = metrics.Precision()
re = metrics.Precision()
acc = metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

img = cv2.imread('aithingy\\humanart1.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show() 

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

np.expand_dims(resize, 0)
yhat = model.predict(np.expand_dims(resize/255, 0))     #the 
print(yhat)

#on startup, do a get request that launches this code. Then, each get req can do a model.predict