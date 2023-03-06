import numpy as np
import keras.api._v2.keras as keras

from configs import *
from utils import loadJSON
from data_utils.dataloader import DataLoader
from data_utils.dataset import Dataset
from inference import get_final_preds, save_batch_heatmaps, save_batch_image_with_keypoints


class PredictCallback(keras.callbacks.Callback):
    def __init__(self, step=10):
        self.step = step

        test_label = loadJSON(TRAIN_LABEL)
        test_data_loader = DataLoader(TEST_PATH)
        test_data = test_data_loader.get_data() # fnames
        test_dataset_loader = Dataset(TEST_PATH, test_data, test_label)

        self.test_fname_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size=TEST_BATCH_SIZE)
        self.test_ds = test_dataset_loader.get_dataset(resize=IMG_SIZE, training=False).batch(batch_size=TEST_BATCH_SIZE)
        

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.step == 0:
            for test_fnames, test_data in zip(self.test_fname_ds, self.test_ds):
                fnames = [ f'epoch_{epoch}_{x.decode()}' for x in test_fnames.numpy() ] # fname의 type이 byte여서 decode를 이용해 string으로 변환
                test_imgs = test_data[0].numpy()
                batch_heatmaps = self.model.predict(test_imgs) # model output shape = [batch_size, w, h, num_keypoints]
                batch_heatmaps = np.transpose(batch_heatmaps, (0, 3, 1, 2)) # transpose to [batch_size, num_keypoints, w, h]

                keypoints_preds = get_final_preds(batch_heatmaps)
                save_batch_heatmaps(test_imgs, batch_heatmaps, fnames)
                save_batch_image_with_keypoints(fnames, test_imgs, keypoints_preds)
            print(f'    [PredictCallback] Epoch {epoch}: inference test imgs saved.')
        else:
            pass
