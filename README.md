# vgg16_ismContest
Training ConvNet for Invasive Species Monitoring data set contest on Kaggle : 
Dataset is available at : https://www.kaggle.com/c/invasive-species-monitoring

# Prerequisites
- Keras
- Tensorflow
- Numpy
- PIL
- sklearn
- scipy 

# How to use
first run the preprocessing script with corrected path to dataset :
```
train_path = './DATA/train/'
test_path = './DATA/test/'
train_lable_file = './DATA/train_labels.csv'
```
the run the Train script to load and train the model on the dataset.

# what does this code do
The train script initializes the VGG16 arc Convent with weights of trained model on imagenet then it finetunes the weights to fit the model to predict the invasive species in the training images.
```
 img_input = Input(shape=(224,224,3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    Lx = Dense(4096, activation='relu', name='fc2')(x)
    
    
    x = Dense(128, activation='relu', name='fc3')(Lx)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(img_input, x, name='vgg16')
    base_model = Model(img_input,Lx,name='baseVgg16')

```

you could set the weights to be not trained and just finetune the last layer of model:
```
for layer in base_model.layers:
    layer.trainable = False 
```


