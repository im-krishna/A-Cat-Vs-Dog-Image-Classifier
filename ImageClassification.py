#importing the libraries
import tensorflow as tf #deep learnig library developed by google
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

#preprocessing and augmenting the training data  
train_datagen = ImageDataGenerator(
    #feature scalling 
    rescale=1./255,#dividing each pixel by 255 so that each pixel lie between 0 and 1
    #image transformation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
 

training_set = train_datagen.flow_from_directory(
    'dataset/training_set', #location of folder
    target_size = (64, 64),#final size of image fed into convolution network
    batch_size = 32, #32 set of image in each batch
    class_mode='binary') #binary outcome 


#preprocessing the testing data
test_datagen = ImageDataGenerator(rescale=1./255)
#applying feature scaling to test set only 

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


#developing the cnn model
cnn = tf.keras.models.Sequential()


#applying first layer of convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,#32 filters are applied
                               kernel_size=3,#each have a kernel size of 3
                               activation='relu', #activaitoj function used is rectified linear unit
                               input_shape=[64,64,3]))#with image size as 64x64 and 3 beacuse we are using rgb image



#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),#size of the kernel of pool 
                                  strides=2))#movement of kernel to the right by 2 units


#2nd layer of convolution and pooling
cnn.add(tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               activation='relu', 
                              ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),
                                  strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())


#finalying applying the artifical neural network
cnn.add(tf.keras.layers.Dense(units =128, #no of hidden neurons
                              activation='relu'))



#final layer or the output layer
cnn.add(tf.keras.layers.Dense(units =1, #no of final neurons is 1 binary classification
                              activation='sigmoid'))


#compiling the cnn model
cnn.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])


#training the data
cnn.fit(x = training_set,
        validation_data= test_set,
        epochs=25)



#now making a single prediction for image
test_image = image.load.img('dataset\single_prediction\cat_or_dog_1.jpg',
                            target_size = (64, 64)) # target_size image
test_image = image.img_to_array(test_image) 
test_image = np.expand_dims(test_image, axis = 0)
#before [1,2]
#after [[1,2],another_image]
#we have to add an extra dimension to the image because we have created batches of image 
#and each batch has 32 images [image,image,.....32]
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat' 
print(prediction)
