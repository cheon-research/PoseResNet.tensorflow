import numpy as np
import tensorflow as tf

from configs import *


class Dataset:
    def __init__(self, path, data, label_dict) -> None:
        self.path = path
        self.label_dict = label_dict
        self.ds = self._init_ds(data)
        self.dataset_size = len(data)
        self.transforms = None
        self.img_size = (IMG_SIZE, IMG_SIZE)
        self.heatmap_size = (HEATMAP_SIZE, HEATMAP_SIZE)
        self.sigma = 2

    def _init_label_map(self, fname):
        keypoints = np.array(self.label_dict[fname])
        keypoints = (keypoints * IMG_SIZE).astype(dtype=np.int32)
        assert keypoints.shape == (4, 2), f'Label shape error! ({fname})'
        #label = np.reshape(label * IMG_SIZE, -1)
        # [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
        # => [ x1, y1, x2, y2, x3, y3, x4, y4 ]
        return keypoints

    def _init_img_map(self, fname, keypoints):
        _f = tf.io.read_file(self.path + fname)
        img = tf.image.decode_jpeg(_f, channels=3)
        return img, keypoints

    def _init_ds(self, data):
        ds_labels = list(map(self._init_label_map, data))
        ds = tf.data.Dataset.from_tensor_slices((data, ds_labels))
        ds = ds.map(self._init_img_map)
        return ds

    def _augmentation_func(self, img, keypoints):
        aug = self.transforms(image=img, keypoints=keypoints)
        aug_img = aug['image']
        aug_keypoints = aug['keypoints'] # [ ( , ), ( , ) ...] (type: list)
        return aug_img, aug_keypoints

    def _resize_func(self, img, keypoints):
        img = tf.cast(tf.image.resize(img, [self.resize, self.resize]), tf.uint8)
        return img, keypoints

    def _img_normalize_func(self, img, heatmap, keypoints):
        img = tf.cast(img, tf.float32) / 255
        return img, heatmap, keypoints

    def _set_shape_func(self, img, heatmap, keypoints):
        img.set_shape(IMG_SHAPE)
        heatmap.set_shape(HEATMAP_SHAPE)
        keypoints.set_shape(LABEL_SHAPE)
        return img, heatmap, keypoints
    
    def set_transforms(self, transforms):
        self.transforms = transforms

    def get_dataset_size(self):
        '''
        dataset 전체 사이즈(batch로 나누기 전)
        '''
        return self.dataset_size


    def get_dataset(self, transforms=None, resize=None, training=True):
        '''
        transforms가 있으면 augmentation 적용한 데이터를 반환
        '''
        ds = self.ds
        if resize:
            self.resize = resize
            ds = ds.map(self._resize_func)
        if transforms:
            self.transforms = transforms
            ds = ds.map(lambda img, keypoints: tf.numpy_function(func=self._augmentation_func, inp=[img, keypoints], Tout=((tf.uint8, tf.int32))))
        ds = ds.map(lambda img, keypoints: tf.numpy_function(func=self._generate_heatmap, inp=[img, keypoints], Tout=((tf.uint8, tf.float32, tf.int32))))
        ds = ds.map(self._img_normalize_func)
        ds = ds.map(self._set_shape_func)

        if training:
            return ds.prefetch(AUTOTUNE).shuffle(buffer_size=self.dataset_size)
        else:
            return ds.prefetch(AUTOTUNE)

    def _generate_heatmap(self, img, keypoints):
        '''
        https://github.com/HRNet/deep-high-resolution-net.pytorch/blob/master/lib/dataset/JointsDataset.py#L233 -> generate_target

        Args:
            img(np.array)
            keypoints(list): [num_keypoints, 2]
        Returns:
            img(np.array)
            heatmap(np.array)
            keypoints(list)
        '''
        self.num_keypoints = len(keypoints)
        target_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)
        heatmap_shape = (self.num_keypoints, self.heatmap_size[0], self.heatmap_size[1])
        heatmap = np.zeros(heatmap_shape, dtype=np.float32)
        tmp_size = self.sigma * 3

        for keypoints_id in range(self.num_keypoints):
            #feat_size = (self.img_size[0]-1.0) / (self.heatmap_size[0]-1.0)
            feat_size = self.img_size[0] / self.heatmap_size[0]
            feat_stride = (feat_size, feat_size)
            #feat_stride = self.img_size / self.heatmap_size
            mu_x = int(keypoints[keypoints_id][0] / feat_stride[0] + 0.5)
            mu_y = int(keypoints[keypoints_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[keypoints_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            w = target_weight[keypoints_id]
            if w > 0.5:
                heatmap[keypoints_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        #heatmap = heatmap.reshape((-1, 4, self.heatmap_size[0], self.heatmap_size[0]))
        return img, heatmap, keypoints
