'''
https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/loss.py
'''

import tensorflow as tf
import keras.api._v2.keras as keras


class KeypointMSELoss(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.criterion = keras.losses.MeanSquaredError(reduction, name)

    def call(self, y_true, y_pred):
        num_keypoints = y_true.shape[1]
        img_size = y_true.shape[2]

        heatmap_trues = y_true
        # model의 output에서 마지막 dimension이 channel(keypoints)여서 ground truth와 연산을 위해 transpose
        heatmap_preds = tf.transpose(y_pred, (0, 3, 1, 2))
        heatmap_trues = tf.reshape(heatmap_trues, [-1, num_keypoints, img_size*img_size])
        heatmap_preds = tf.reshape(heatmap_preds, [-1, num_keypoints, img_size*img_size])

        # axis=1을 기준으로 num_keypoints 개수로 tensor를 나눠서 list로 반환
        heatmap_trues = tf.split(heatmap_trues, num_or_size_splits=num_keypoints, axis=1)
        heatmap_preds = tf.split(heatmap_preds, num_or_size_splits=num_keypoints, axis=1)
        loss = 0

        for true, pred in zip(heatmap_trues, heatmap_preds):
            #print(true.shape, pred.shape)
            true = tf.squeeze(true)
            pred = tf.squeeze(pred)
            #print('after', true.shape, pred.shape)
            loss += 0.5 * self.criterion(true, pred)
            #loss += 0.5 * tf.reduce_mean(tf.square(pred - true), axis=-1)
        return loss / num_keypoints
