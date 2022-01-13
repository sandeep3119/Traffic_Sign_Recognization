# Traffic_Sign_Recognization
With the use of Kaggle Traffic Sign Dataset, A CNN model is trained to recognize traffic sign with more than 43 classes.

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


