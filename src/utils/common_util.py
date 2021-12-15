import yaml

def read_config(config_path):
    with open(config_path) as config:
        content=yaml.safe_load(config)
    return content