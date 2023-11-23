import tensorflow as tf
from tensorflow.keras.layers import Input

# def relu(x):
#     return tf.keras.nn.relu(x)

# def upsample_and_sum(x1, x2, output_channels, in_channels):
#     pool_size = 2
#     deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
#     deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
#     deconv_output = tf.keras.layers.Add()([deconv, x2])
#     return deconv_output

def upsample_and_sum(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=(pool_size, pool_size), strides=(pool_size, pool_size), padding='same')(x1)
    deconv_output = tf.keras.layers.Add()([deconv, x2])
    return deconv_output



def deep_flexisp_1(input_shape):
    input = Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(input)
    res_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(conv1)
    res_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_conv1)
    res_block1=tf.keras.layers.Add()([conv1, res_conv1])

    pool2 = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='same')(res_block1)
    res_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(pool2)
    res_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_conv2)
    res_block2=tf.keras.layers.Add()([pool2, res_conv2])

    pool3 = tf.keras.layers.AveragePooling2D(pool_size=[2, 2], strides=2, padding='same')(res_block2)
    res_conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(pool3)
    res_conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_conv3)
    res_block3=tf.keras.layers.Add()([pool3, res_conv3])

    deconv1 = upsample_and_sum(res_block3, res_block2, 64, 64)
    

    conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(deconv1)
    res_conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(conv4)
    res_conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_conv4)
    res_block4=tf.keras.layers.Add()([conv4, res_conv4])

    deconv2 = upsample_and_sum(res_block4, res_block1, 64, 64)

    conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(deconv2)
    res_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(conv5)
    res_conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_conv5)
    res_block5=tf.keras.layers.Add()([conv5, res_conv5])

    conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=(1,1), padding='same', activation='relu')(res_block5)
    conv7 = tf.keras.layers.Conv2D(filters=3, kernel_size=[3, 3], strides=(1,1), padding='same', activation=None)(conv6)

    model = tf.keras.models.Model(inputs=input, outputs=conv7)    

    return model