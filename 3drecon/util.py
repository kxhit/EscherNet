import importlib
import pickle
import numpy as np
import cv2

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def output_points(fn,pts,colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi,0])} {int(colors[pi,1])} {int(colors[pi,2])}')
            f.write('\n')

def concat_images(img0,img1,vert=False):
    if not vert:
        h0,h1=img0.shape[0],img1.shape[0],
        if h0<h1: img0=cv2.copyMakeBorder(img0,0,h1-h0,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        if h1<h0: img1=cv2.copyMakeBorder(img1,0,h0-h1,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0,w1=img0.shape[1],img1.shape[1]
        if w0<w1: img0=cv2.copyMakeBorder(img0,0,0,0,w1-w0,borderType=cv2.BORDER_CONSTANT,value=0)
        if w1<w0: img1=cv2.copyMakeBorder(img1,0,0,0,w0-w1,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img

def concat_images_list(*args,vert=False):
    if len(args)==1: return args[0]
    img_out=args[0]
    for img in args[1:]:
        img_out=concat_images(img_out,img,vert)
    return img_out