import json
import os

def get(key, default=None):
    username = os.getusername()
    with open(f'config.json') as config_file:
        config = json.load(config_file)
    
    # first, check if config[username] exists
    if username in config:
        if key in config[username]:
            return config[username][key]
    
    # if not, check if key exists in config['default']
    if 'default' in config:
        if key in config['default']:
            return config['default'][key]
    
    # if not, and default is not None, return default
    if default is not None:
        return default

    # Otherwise, raise an error
    raise KeyError(f'Key {key} not found in config file')