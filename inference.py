import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math

from configs import *
from utils import draw_points, loadJSON
from data_utils.dataloader import DataLoader
from data_utils.dataset import Dataset
from train_utils.loss import KeypointMSELoss


def get_max_preds(batch_heatmaps):
    '''
    https://github.com/HRNet/deep-high-resolution-net.pytorch/blob/master/lib/core/inference.py

    get predictions from score maps

    Args:
        heatmaps(np.ndarray): [batch_size, num_keypoints, height, width]
    Returns:
        preds(np.ndarray)
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_keypoints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_keypoints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_keypoints, 1))
    idx = idx.reshape((batch_size, num_keypoints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds


def get_final_preds(batch_heatmaps):
    coords = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    #if config.TEST.POST_PROCESS:
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array([hm[py][px+1] - hm[py][px-1],
                                    hm[py+1][px]-hm[py-1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    return preds


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    https://github.com/HRNet/deep-high-resolution-net.pytorch/blob/master/lib/utils/vis.py#L54

    batch_image: [batch_size, channel, height, width] or [batch_size, height, width, channel]
    batch_heatmaps: [batch_size, num_keypoints, height, width]
    file_name: saved file name
    '''

    if batch_image.shape[-1] == 3:
        batch_image = np.transpose(batch_image, (0, 3, 1, 2)) 

    if normalize:
        #batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())
        batch_image = (batch_image - min) / (max - min + 1e-5)


    batch_size = batch_heatmaps.shape[0]
    num_keypoints = batch_heatmaps.shape[1]
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    grid_image = np.zeros((batch_size*heatmap_height,
                        (num_keypoints+1)*heatmap_width,
                        3),
                        dtype=np.uint8)

    preds = get_max_preds(batch_heatmaps)

    for i in range(batch_size):
        image = np.clip(batch_image[i]*255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
        heatmaps = np.clip(batch_heatmaps[i]*255, 0, 255).astype(np.uint8)

        resized_image = cv2.resize(image,
                                (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_keypoints):
            cv2.circle(resized_image,
                    (int(preds[i][j][0]), int(preds[i][j][1])),
                    1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                    (int(preds[i][j][0]), int(preds[i][j][1])),
                    1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'../results/heatmaps/heatmap_{file_name[0]}', grid_image)


def save_batch_image_with_keypoints(fnames, batch_images, batch_keypoints):
    #scale = IMG_SIZE / HEATMAP_SIZE
    batch_images = (batch_images * 255).astype(np.uint8)
    batch_keypoints = (batch_keypoints / HEATMAP_SIZE * IMG_SIZE).astype(np.int32)

    for fname, image, keypoints in zip(fnames, batch_images, batch_keypoints):
        img = draw_points(image, points=keypoints, points_labels=POINTS_LABELS)
        cv2.imwrite(f'../results/preds/{fname}', img) 
    #print('[save_batch_image_with_keypoints] done')


def main(model_path):
    test_label = loadJSON(TEST_LABEL)
    test_data_loader = DataLoader(TEST_PATH)
    test_data = test_data_loader.get_data() # file names
    test_dataset_loader = Dataset(TEST_PATH, test_data, test_label)
    test_ds = test_dataset_loader.get_dataset(resize=IMG_SIZE, training=False).batch(batch_size=TEST_BATCH_SIZE)
    test_fname_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size=TEST_BATCH_SIZE)

    model = keras.models.load_model(model_path)
    loss_func = KeypointMSELoss()

    for test_fnames, test_data in zip(test_fname_ds, test_ds):
        fnames = [ x.decode() for x in test_fnames.numpy() ] # fname의 type이 byte여서 decode를 이용해 string으로 변환
        test_imgs = test_data[0].numpy()
        batch_heatmaps = model.predict(test_imgs) # model output shape = [batch_size, w, h, num_keypoints]
        batch_heatmaps = np.transpose(batch_heatmaps, (0, 3, 1, 2)) # transpose to [batch_size, num_keypoints, w, h]

        keypoints_preds = get_final_preds(batch_heatmaps)
        save_batch_heatmaps(test_imgs, batch_heatmaps, fnames)
        save_batch_image_with_keypoints(fnames, test_imgs, keypoints_preds)
    
    test_loss = .0
    for test_imgs, test_heatmaps, test_labels in test_ds:
        pred_heatmaps = model.predict(test_imgs)
        loss = loss_func(test_heatmaps, pred_heatmaps).numpy()
        test_loss += loss
    print('TEST LOSS: {:.5f}'.format(test_loss))