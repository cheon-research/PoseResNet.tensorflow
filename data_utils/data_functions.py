import tensorflow as tf

import glob

from configs import *


def concat_datasets(ds: tf.data.Dataset, ds_ls: list) -> tf.data.Dataset:
    for _ds in ds_ls:
        ds = ds.concatenate(_ds)
    ds_size = ds.cardinality()
    return ds.shuffle(buffer_size=ds_size, reshuffle_each_iteration=True)

def load_data(dir: str, extension: str='jpg') -> list:
    fpath = f'{dir}/*.{extension}'
    img_ls = glob.glob(fpath)
    
    assert len(img_ls) > 0, f'No data to load at {dir}'
    
    img_ls = list(map(lambda x: x.split('/')[-1], img_ls))
    return img_ls
