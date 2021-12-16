from sklearn import metrics
from utils.image_processing_util import arrange_data
from utils.model_util import build_model,test_model
from utils.common_util import read_config
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mlflow.keras
import mlflow.tensorflow
import mlflow

def training(config_path):
    mlflow.set_tracking_uri('http://127.0.0.1:1234')
    mlflow.set_experiment('Cnn_architechture')
    mlflow.tensorflow.autolog()
    config=read_config(config_path)
    train_data_folder=config['data_params']['train_data_path']
    data,labels=arrange_data(train_data_folder)
    split_ratio=config['data_params']['split_ratio']
    X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=split_ratio,random_state=42)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    y_train=to_categorical(y_train,43)
    y_test=to_categorical(y_test,43)
    model=build_model(X_train)
    LOSS=config['model_params']['loss']
    OPTIMIZER=config['model_params']['optimizer']
    METRICS=config['model_params']['metrics']
    EPOCHS=config['model_params']['epochs']
    BATCH_SIZE=config['model_params']['batch_size']
    model.compile(loss=LOSS,optimizer=OPTIMIZER,metrics=METRICS)
    history=model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(X_test,y_test))
    test_file=config['data_params']['test_path']
    score = test_model(test_file,model)
    print(f"Test_Accuracy:{score}")
   




if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config','-c',default='config.yaml')
    parsed_arg=args.parse_args()
    training(parsed_arg.config)
