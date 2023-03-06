import tensorflow as tf
from tensorflow import keras

import datetime

from configs import *
from model import pose_resnet


import utils
from data_utils.dataloader import DataLoader
from data_utils.dataset import Dataset

from train_utils import train_functions
from train_utils.loss import KeypointMSELoss
from callbacks.predict import PredictCallback
from callbacks.save_model import SaveModelCallback


now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
now_str = now.strftime('%y%m%d%H%M%S')

# prepare dataset
train_label = utils.loadJSON(TRAIN_LABEL)
train_data_loader = DataLoader(TRAIN_PATH)
train_data = train_data_loader.get_data()

train_dataset_loader = Dataset(TRAIN_PATH, train_data, train_label)
train_ds = train_dataset_loader.get_dataset(resize=IMG_SIZE)
train_size = train_dataset_loader.get_dataset_size()

val_size = int(train_size * VAL_RATIO)
train_ds = train_ds.skip(val_size)
val_ds = train_ds.take(val_size)

train_len, val_len = train_ds.cardinality(), val_ds.cardinality()
print(f'Dataset #train: {train_size}, #train+aug: {train_len}, #val_len: {val_len}')

train_ds = train_ds.batch(batch_size=BATCH_SIZE)
val_ds = val_ds.batch(batch_size=BATCH_SIZE)

# callbacks
cb_lr_scheduler = keras.callbacks.LearningRateScheduler(train_functions.lr_scheduler)
cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
cb_logger = keras.callbacks.CSVLogger(f"{RESULT_PATH}/logs/train_log_{now_str}.csv", separator='\t')
cb_inference = PredictCallback(step=10)
cb_save_model = SaveModelCallback(path=f"{RESULT_PATH}/models/", now_str=now_str)
callbacks = [cb_lr_scheduler, cb_early_stop, cb_logger, cb_inference, cb_save_model]

# set-up model
model = pose_resnet.PoseResNet(input_shape=IMG_SHAPE)
optim = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_func = KeypointMSELoss()
model.compile(optimizer=optim, loss=loss_func)
train_functions.plot_model(model, f"{RESULT_PATH}/model_{now_str}.png")

# train
history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCH, callbacks=callbacks)
train_functions.plot_loss(history, start_epoch=1, now_str=now_str)
