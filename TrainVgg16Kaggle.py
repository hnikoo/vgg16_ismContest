from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam
import numpy as np
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
    
# Import data
from preprocess import load_data
Xtrain,ytrain,Xval,yval,Xtest,ytest = load_data()
Xtrain = Xtrain.astype('float') / 255.0
Xtest = Xtest.astype('float') / 255.0
Xval = Xval.astype('float') / 255.0



def VGG_16(weights_path=None):
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
    
    #for layer in base_model.layers:
    #    layer.trainable = False    
        
    vgg = VGG16(include_top=True,input_shape=(224,224,3))
    vggW = vgg.get_weights()
    
    mW = model.get_weights()
    
    newW = vggW[:-2] + mW[-4:]
    
    model.set_weights(newW)
    
    
    
    print model.summary()
    return model



if __name__ == "__main__":
    model = VGG_16()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)#Adam(lr=0.0001, beta_1=0.5)#Adadelta()#
    #model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy']) 
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
    mcp = ModelCheckpoint(filepath='./best_model.hdf5', verbose=1,monitor='val_loss',save_best_only=True)
    
    batch_size = 16
    epochs = 50
    
    history = model.fit(Xtrain, ytrain[:,0],
                        shuffle=True,
                        batch_size=8, 
                        nb_epoch=15,
                        verbose=1, callbacks=[mcp],
                        validation_data=(Xval,yval[:,0]))
    print np.argmax(out)