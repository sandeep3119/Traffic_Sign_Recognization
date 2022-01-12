from src.utils.common_util import read_config
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import os


def log_production_model(config_path):
    config=read_config(config_path)

    mlflow_config=config['mlflow_config']
    model_name=mlflow_config['registered_model_name']
    remote_server_uri=mlflow_config['remote_server_uri']
    mlflow.set_tracking_uri(remote_server_uri)
    runs=mlflow.search_runs(experiment_ids=1)
    highest=runs['metrics.test_accuracy'].sort_values()[0]
    highest_run_id=runs[runs['metrics.test_accuracy']==highest]['run_id'][0]
    client=MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv=dict(mv)
        if mv['run_id']==highest_run_id:
            current_verison=mv['version']
            logged_model=mv['source']
            pprint(mv,indent=4)
            client.transition_model_version_stage(name=model_name,
            version=current_verison,
            stage='Production')
        else:
            current_verison=mv['version']
            client.transition_model_version_stage(
                name=model_name,
                version=current_verison,
                stage='Staging'
            )
    loaded_model=mlflow.keras.load_model(logged_model)
    model_path=config['prediction_service']['model_dir'] 
    loaded_model.save(model_path)


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config','-c',default='config.yaml')
    parsed_arg=args.parse_args()
    log_production_model(parsed_arg.config)

