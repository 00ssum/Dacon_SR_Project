# layernorm추가
# activation: mish -> leakymish로 변경

import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow_addons as tfa

def upsample_and_sum(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=(pool_size, pool_size), strides=(pool_size, pool_size), padding='same')(x1)
    deconv_output = tf.keras.layers.Add()([deconv, x2])
    return deconv_output

def residual_block(input):
    res_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(input)
    
    res_conv1 = tf.keras.layers.LayerNormalization()(res_conv1)
    res_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(res_conv1)
    res_conv2 = tf.keras.layers.LayerNormalization()(res_conv2)
    return res_conv2


def deep_flexisp_1_2(input_shape):
    input = Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(input)
    conv1 = tf.keras.layers.LayerNormalization()(conv1)
    res_conv1=residual_block(conv1)
    res_block1=tf.keras.layers.Add()([conv1, res_conv1])

    pool2 = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='same')(res_block1)
    res_conv2=residual_block(pool2)
    res_block2=tf.keras.layers.Add()([pool2, res_conv2])

    pool3 = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='same')(res_block2)
    res_conv3=residual_block(pool3)
    res_block3=tf.keras.layers.Add()([pool3, res_conv3])

    deconv1 = upsample_and_sum(res_block3, res_block2, 64, 64)
    
    
    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(deconv1)
    conv4 = tf.keras.layers.LayerNormalization()(conv4)
    res_conv4=residual_block(conv4)
    res_block4=tf.keras.layers.Add()([conv4, res_conv4])

    deconv2 = upsample_and_sum(res_block4, res_block1, 64, 64)


    conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(deconv2)
    conv5 = tf.keras.layers.LayerNormalization()(conv5)
    res_conv5=residual_block(conv5)
    res_block5=tf.keras.layers.Add()([conv5, res_conv5])


    conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation=tfa.activations.mish)(res_block5)
    conv6 = tf.keras.layers.LayerNormalization()(conv6)
    
    conv6 = tf.keras.layers.Dropout(0.05)(conv6)
    conv7 = tf.keras.layers.Conv2D(filters=3, kernel_size=[3, 3], strides=(1,1), padding='same', activation='sigmoid')(conv6)

    model = tf.keras.models.Model(inputs=input, outputs=conv7)  

    return model