import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model

def psnr_loss(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1.0)

def dssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    return (1.0-ssim)/2.0

def ssim_loss(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)

class vggModel(object):
    def __init__(self, image_shape=(32, 32, 3)):
        self.image_shape = image_shape
        self.vgg = VGG16(include_top=False, weights='imagenet', input_shape=self.image_shape)
        self.selectedLayers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
        self.selectedOutputs = [self.vgg.get_layer(i).output for i in self.selectedLayers]
        self.loss_model = Model(inputs=self.vgg.input, outputs=self.selectedOutputs)
        self.loss_model.trainable = False
        self.mse = MeanSquaredError()

    def perceptual_loss(self, y_true, y_pred):
        loss_value = 0
        for i in range(0,4):
            loss_value += self.mse(self.loss_model(y_true)[i], self.loss_model(y_pred)[i])
        return loss_value

class efficientnetModel(object):
    def __init__(self, image_shape=(None, None, 3)):
        self.image_shape = image_shape
        self.efficient = EfficientNetB2(include_top=False, weights='imagenet', input_shape=self.image_shape)
        self.selectedLayers = ['block2a_expand_activation','block3a_expand_activation','block4a_expand_activation']
        self.selectedOutputs = [self.efficient.get_layer(i).output for i in self.selectedLayers]
        self.loss_model = Model(inputs=self.efficient.input, outputs=self.selectedOutputs)
        self.loss_model.trainable = False
        self.mse = MeanSquaredError()

    def perceptual_loss(self, y_true, y_pred):
        loss_value = 0
        y_true = tf.math.scalar_mul(255.0, y_true)      # EfficientNet takes values between 0-255
        y_pred = tf.math.scalar_mul(255.0, y_pred)
        for i in range(0,3):
            loss_value += self.mse(self.loss_model(y_true)[i], self.loss_model(y_pred)[i])
        return loss_value

class DenseNetModel(object):
    def __init__(self, image_shape=(None, None, 3)):
        self.image_shape = image_shape
        self.densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=self.image_shape)
        self.selectedLayers = ['conv1/relu','pool2_conv','pool3_conv']
        self.selectedOutputs = [self.densenet.get_layer(i).output for i in self.selectedLayers]
        self.loss_model = Model(inputs=self.densenet.input, outputs=self.selectedOutputs)
        self.loss_model.trainable = False
        self.mse = MeanSquaredError()

    def perceptual_loss(self, y_true, y_pred):
        loss_value = 0
        for i in range(0,3):
            loss_value += self.mse(self.loss_model(y_true)[i], self.loss_model(y_pred)[i])
        return loss_value

class ResNet50V2Model(object):
    def __init__(self, image_shape=(None, None, 3)):
        self.image_shape = image_shape
        self.resnet = ResNet50V2(include_top=False, weights='imagenet', input_shape=self.image_shape)
        self.selectedLayers = ['conv1_conv','conv2_block2_out','conv3_block3_out']
        self.selectedOutputs = [self.resnet.get_layer(i).output for i in self.selectedLayers]
        self.loss_model = Model(inputs=self.resnet.input, outputs=self.selectedOutputs)
        self.loss_model.trainable = False
        self.mse = MeanSquaredError()

    def perceptual_loss(self, y_true, y_pred):
        loss_value = 0
        for i in range(0,3):
            loss_value += self.mse(self.loss_model((y_true*2.0)-1.0)[i], self.loss_model((y_pred*2.0)-1.0)[i])
        return loss_value

if __name__ =='__main__':
    # vgg = vggModel()
    # densenet = DenseNetModel()
    # efficient = efficientnetModel()
    resnet = ResNet50V2Model()
