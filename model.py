'''
https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
'''

from tensorflow import keras
from tensorflow.keras import layers

from configs import *


class PoseResNet(keras.Model):
    def __init__(self, input_shape, backbon_trainable=False) -> None:
        super().__init__()

        self.backbone = keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        self.backbone.trainable = backbon_trainable

        self.deconvs = self._make_deconv_layer(
            POSE_RESNET.NUM_DECONV_LAYERS, # 3
            POSE_RESNET.NUM_DECONV_FILTERS, # [256, 256, 256]
            POSE_RESNET.NUM_DECONV_KERNELS, # [4, 4, 4]
        )
        self.final_layer = layers.Conv2D(
            filters=NUM_KEYPOINTS,
            kernel_size=POSE_RESNET.FINAL_CONV_KERNEL,
            strides=1,
            padding='same' if POSE_RESNET.FINAL_CONV_KERNEL == 3 else 'valid'
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_kernels)'

        deconvs = keras.Sequential(name='Deconvs')
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]

            deconvs.add(layers.Conv2DTranspose(
                filters=planes,
                kernel_size=kernel,
                strides=2,
                padding='same', #if padding == 1 else 'valid',
                #output_padding=output_padding,
                use_bias=POSE_RESNET.DECONV_WITH_BIAS
            ))
            deconvs.add(layers.BatchNormalization(momentum=BN_MOMENTUM))
            deconvs.add(layers.ReLU())
        return deconvs

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.deconvs(x)
        x = self.final_layer(x)
        return x

    def build_graph(self):
        # https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
        inputs = layers.Input(shape=IMG_SHAPE, name='InputImage')
        return keras.Model(inputs=inputs, outputs=self.call(inputs))
