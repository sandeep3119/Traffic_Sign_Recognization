# Traffic_Sign_Recognization
With the use of Kaggle Traffic Sign Dataset, A CNN model is trained to recognize traffic sign with more than 43 classes.

This project uses MLFLOW for logging purpose and contains github worflows aswell for ci-cd pipeline.
Project is deployed on Heroku.
To deploy the project on your heroku server configure
    HEROKU_API_TOKEN: ${{ secrets.HEROKU_API_TOKEN }}
    HEROKU_APP_NAME: ${{ secrets.HEROKU_APP_NAME }} 
on your github account.

### Data set Used
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


### WebApp Demo At
https://trafficsign-recognizer.herokuapp.com/

## To run mlflow ui
Run Mlflow before running any training. This will setup mlflow db and experiment for us.
```mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns```

## To Edit the model architecture
Edit src/utils/model_uti.py

## To edit just model parameters
Edit config.yaml

## Run Training after editing
```python src/training.py```
Each training will be logged to Mlflow with parameter,architecture,metrics and model

After some different model architectures and paramter, try to log best model to production.

## To deploy a model into production based on some score
```python src/log_production_model.py```

## To run the webapp
```python app.py```


## Present Model Architecture

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
    
    With epochs=20, optimizer=adam and batch_size=32 got an accuracy of *0.95*
