import os
import cv2
import yaml
import numpy as np

COLOR_MASK = {0:(0,0,0), 1:(128,0,0)}
COLOR_OVERLAY = {0:(0,0,0), 1:(0,255,0)}

def mask2rgb(mask, size=(256,256), is_overlay=False):
    mask = np.array(mask, dtype=np.uint8)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)

    for i in np.unique(mask):
        if is_overlay:
            rgb[mask==i] = COLOR_OVERLAY[i]
        else:
            rgb[mask==i] = COLOR_MASK[i]

    return rgb

def checkDir(path, auto_increment=False):
    if os.path.exists(path):
        if auto_increment:
            for cnt in range(1,9999):
                _path = path[:-1]+"_{:02d}/".format(cnt)
                if not os.path.exists(_path):
                    os.makedirs(_path)
                    return _path
        else:
            return path
    else:
        os.makedirs(path)
        return path

def getConfig(config_path=None):
    if not config_path:
        config_path = "./config/config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config