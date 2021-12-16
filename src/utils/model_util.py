from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.image_processing_util import load_image
import numpy as np
from sklearn.metrics import accuracy_score

def build_model(X_train):
    layers=[
        Conv2D(filters=32,kernel_size=(5,5),activation='relu',input_shape=X_train.shape[1:]),
        Conv2D(filters=32,kernel_size=(5,5),activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(rate=0.2),
        Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(rate=0.2),
        Flatten(),
        Dense(units=256,activation='relu'),
        Dropout(rate=0.4),
        Dense(units=43,activation='softmax')
    ]
    model=Sequential(layers=layers)
    return model

def save_model(model,path):
    model_path=os.makedirs(path,'model.h5')
    model.save(model_path)
    return model_path
    

def test_model(test_data,model):
    y_test=pd.read_csv('Test.csv')
    actual=y_test['ClassId'].values
    images=y_test['Path'].values
    data=[]
    for img in images:
        image=load_image(img)
        data.append(image)
    X_test=np.array(data)
    pred=model.predict_classes(X_test)
    return accuracy_score(actual,pred)


