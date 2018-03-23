import os;
import skimage.data
from skimage import io
import skimage.transform as T
import skimage.util as util
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50

import numpy as np

from od_data_gen import DataGen

DATAPATH = 'C:\\data\\VOCdevkit\\VOC2007\\'
data_gen = DataGen('trainval', DATAPATH,200)



# Load ResNet50 architecture & its weights
model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()
model = Model(input=model.input,output=model.layers[-1].output)
model.compile(loss='binary_crossentropy', optimizer=SGD(0.5, momentum=0.9), metrics=['binary_accuracy'])


data_set = data_gen.DataSet

print(data_set)

# Initilisation des matrices contenant les Deep Features et les labels
X_train =[]
Y_train =[]


k=1
for i in data_set:
    try:
        # Pour chaque image, on extrait les images des regions d'entrée X et les labels y
        print("%d - Treating Image N° %06d"%(k,i))
        X,Y = data_gen.generate_xy_from_image(i)
        nb_region = len(Y)
        print("\t %d regions proposed"%nb_region)
        
        # On récupère les Deep Feature par appel à predict
        y_pred = model.predict(X)
        print("\t deep features extracted")
        
        X_train.extend(y_pred)
        Y_train.extend(Y)

        k+=1
    except Exception as e:
        pass

    
l = len(Y_train)
X_train = np.reshape(X_train, (l, 2048))
Y_train = np.reshape(Y_train, (l, 21))


outfile = 'RP_DF_ResNet50_VOC2007_train'
np.savez(outfile, X_train=X_train, Y_train=Y_train)
