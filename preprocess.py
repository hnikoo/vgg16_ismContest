import numpy as np
import os
from numpy import genfromtxt
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.cross_validation import train_test_split
from PIL import Image

train_path = './DATA/train/'
test_path = './DATA/test/'
train_lable_file = './DATA/train_labels.csv'


def Preprocess():
    Y = genfromtxt(train_lable_file,delimiter=',')[1:,1]
    
    listImages = os.listdir(train_path)
    fnames = [int(x[:-4]) for x in listImages]
    fnames.sort()
    
    IMAGES = []
    for imname in fnames:    
        img = misc.imread(train_path+str(imname)+'.jpg',mode='RGB')
        img = misc.imresize(img,size=(224,224))        
        IMAGES.append(np.expand_dims(img,axis=0))
        
    IMAGES = np.concatenate(IMAGES,axis=0)
    
    savemat('./DATA/TrainData.mat',{'X':IMAGES,'Y':Y})
        
    
def load_data():
    mat = loadmat('./DATA/TrainData.mat')
    X = mat['X']
    Y = mat['Y']
    Y = np.concatenate((Y.reshape((-1,1)),1.0 - Y.reshape((-1,1))),axis=1)
    
    idx = np.arange(X.shape[0])
    idx_train,idx_test = train_test_split(idx,test_size=0.2)
    idx_train,idx_val = train_test_split(idx_train,test_size=0.2)
    
    Xtrain = X[idx_train,]
    ytrain = Y[idx_train,]
    Xval = X[idx_val,]
    yval = Y[idx_val,]
    Xtest = X[idx_test,]
    ytest = Y[idx_test,]  
    
    return Xtrain,ytrain,Xval,yval,Xtest,ytest



if __name__ == "__main__":
    Preprocess()