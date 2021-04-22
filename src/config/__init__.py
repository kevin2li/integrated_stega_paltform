import yaml
# modify here to specify configure file you want to use
default_cfg = "/home/kevin2li/wave/myapps/src/config/config.yaml"

def getConfig(filename=default_cfg):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

args = getConfig()
