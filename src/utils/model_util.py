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
    

def test_model(test_data,model):
    file=os.path.join(test_data)
    y_test=pd.read_csv(file)
    actual=y_test['ClassId'].values
    images=y_test['Path'].values
    images=[path.replace('/','\\') for path in images]
    data=[]
    for img in images:
        img_path=os.path.join('Data',img)
        image=load_image(img_path)
        data.append(image)
    X_test=np.array(data)
    pred=np.argmax(model.predict(X_test), axis=-1)
    return accuracy_score(actual,pred)


