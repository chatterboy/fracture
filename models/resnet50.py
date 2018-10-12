import tensorflow as tf
from utils.layers import relu, batch_norm, conv2d, maxpool2d, avgpool2d

def block(name, input, filters, ksz, stride, padding):
    """
    :param name: a string
    :param input: a tensor
    :param filters: an integer, # filters
    :param ksz: an integer, the size of filters
    :param stride: an integer
    :param padding: a string, all string need to be uppercase
    :return:
    """
    with tf.variable_scope(name):
        conv = conv2d('conv', input, filters, ksz, stride, padding)
        bn = batch_norm('bn', conv)
        return relu('relu', bn)

def resnet50(input):
    """
    :param input: a 4-D tensor, NHWC
    :return:
    """
    # conv1, 7x7, 64, stride 2
    conv1 = block('conv1', input, 64, 7, 2, 'SAME')

    # conv2_x, 3x3 max pool, stride 2
    maxpool2 = maxpool2d('pool2', conv1, 3, 2, 'SAME')

    # conv2_1
    # conv2_11, 1x1, 64
    conv2_11 = block('conv2_11', maxpool2, 64, 1, 1, 'VALID')
    # conv2_12, 3x3, 64
    conv2_12 = block('conv2_12', conv2_11, 64, 3, 1, 'SAME')
    # conv2_13, 1x1, 256
    conv2_13 = block('conv2_13', conv2_12, 256, 1, 1, 'VALID')
    r2_1 = tf.add_n([conv2d('pj2', maxpool2, 256, 1, 1, 'VALID'), conv2_13])

    # conv2_2
    # conv2_21, 1x1, 64
    conv2_21 = block('conv2_21', conv2_13, 64, 1, 1, 'VALID')
    # conv2_22, 3x3, 64
    conv2_22 = block('conv2_22', conv2_21, 64, 3, 1, 'SAME')
    # conv2_23, 1x1, 256
    conv2_23 = block('conv2_23', conv2_22, 256, 1, 1, 'VALID')
    r2_2 = tf.add_n([r2_1, conv2_23])

    # conv2_3
    # conv2_31, 1x1, 64
    conv2_31 = block('conv2_31', r2_2, 64, 1, 1, 'VALID')
    # conv2_32, 3x3, 64
    conv2_32 = block('conv2_32', conv2_31, 64, 3, 1, 'SAME')
    # conv2_33, 1x1, 256
    conv2_33 = block('conv2_33', conv2_32, 256, 1, 1, 'VALID')
    r2_3 = tf.add_n([r2_2, conv2_33])

    # conv3_x

    # conv3_1
    # conv3_11, 1x1, 128
    conv3_11 = block('conv3_11', r2_3, 128, 1, 2, 'VALID')
    # conv3_12, 3x3, 128
    conv3_12 = block('conv3_12', conv3_11, 128, 3, 1, 'SAME')
    # conv3_13, 1x1, 512
    conv3_13 = block('conv3_13', conv3_12, 512, 1, 1, 'VALID')
    r3_1 = tf.add_n([conv2d('pj3', r2_3, 512, 1, 2, 'VALID'), conv3_13])

    # conv3_2
    # conv3_21, 1x1, 128
    conv3_21 = block('conv3_21', r3_1, 128, 1, 1, 'VALID')
    # conv3_22, 3x3, 128
    conv3_22 = block('conv3_22', conv3_21, 128, 3, 1, 'SAME')
    # conv3_23, 1x1, 512
    conv3_23 = block('conv3_23', conv3_22, 512, 1, 1, 'VALID')
    r3_2 = tf.add_n([r3_1, conv3_23])

    # conv3_3
    # conv3_31, 1x1, 128
    conv3_31 = block('conv3_31', r3_2, 128, 1, 1, 'VALID')
    # conv3_32, 3x3, 128
    conv3_32 = block('conv3_32', conv3_31, 128, 3, 1, 'SAME')
    # conv3_33, 1x1, 512
    conv3_33 = block('conv3_33', conv3_32, 512, 1, 1, 'VALID')
    r3_3 = tf.add_n([r3_2, conv3_33])

    # conv3_4
    # conv3_41, 1x1, 128
    conv3_41 = block('conv3_41', r3_3, 128, 1, 1, 'VALID')
    # conv3_42, 3x3, 128
    conv3_42 = block('conv3_42', conv3_41, 128, 3, 1, 'SAME')
    # conv3_43, 1x1, 512
    conv3_43 = block('conv3_43', conv3_42, 512, 1, 1, 'VALID')
    r3_4 = tf.add_n([r3_3, conv3_43])

    # conv4_x

    # conv4_1
    # conv4_11, 1x1, 256
    conv4_11 = block('conv4_11', r3_4, 256, 1, 2, 'VALID')
    # conv4_12, 3x3, 256
    conv4_12 = block('conv4_12', conv4_11, 256, 3, 1, 'SAME')
    # conv4_13, 1x1, 1024
    conv4_13 = block('conv4_13', conv4_12, 1024, 1, 1, 'VALID')
    r4_1 = tf.add_n([conv2d('pj4', r3_4, 1024, 1, 2, 'VALID'), conv4_13])

    # conv4_2
    # conv4_21, 1x1, 256
    conv4_21 = block('conv4_21', r4_1, 256, 1, 1, 'VALID')
    # conv4_22, 3x3, 256
    conv4_22 = block('conv4_22', conv4_21, 256, 3, 1, 'SAME')
    # conv4_23, 1x1, 1024
    conv4_23 = block('conv4_23', conv4_22, 1024, 1, 1, 'VALID')
    r4_2 = tf.add_n([r4_1, conv4_23])

    # conv4_3
    # conv4_31, 1x1, 256
    conv4_31 = block('conv4_31', r4_2, 256, 1, 1, 'VALID')
    # conv4_32, 3x3, 256
    conv4_32 = block('conv4_32', conv4_31, 256, 3, 1, 'SAME')
    # conv4_33, 1x1, 1024
    conv4_33 = block('conv4_33', conv4_32, 1024, 1, 1, 'VALID')
    r4_3 = tf.add_n([r4_2, conv4_33])

    # conv4_4
    # conv4_41, 1x1, 256
    conv4_41 = block('conv4_41', r4_3, 256, 1, 1, 'VALID')
    # conv4_42, 3x3, 256
    conv4_42 = block('conv4_42', conv4_41, 256, 3, 1, 'SAME')
    # conv4_43, 1x1, 1024
    conv4_43 = block('conv4_43', conv4_42, 1024, 1, 1, 'VALID')
    r4_4 = tf.add_n([r4_3, conv4_43])

    # conv4_5
    # conv4_51, 1x1, 256
    conv4_51 = block('conv4_51', r4_4, 256, 1, 1, 'VALID')
    # conv4_52, 3x3, 256
    conv4_52 = block('conv4_52', conv4_51, 256, 3, 1, 'SAME')
    # conv4_53, 1x1, 1024
    conv4_53 = block('conv4_53', conv4_52, 1024, 1, 1, 'VALID')
    r4_5 = tf.add_n([r4_4, conv4_53])

    # conv4_6
    # conv4_61, 1x1, 256
    conv4_61 = block('conv4_61', r4_5, 256, 1, 1, 'VALID')
    # conv4_62, 3x3, 256
    conv4_62 = block('conv4_62', conv4_61, 256, 3, 1, 'SAME')
    # conv4_63, 1x1, 1024
    conv4_63 = block('conv4_63', conv4_62, 1024, 1, 1, 'SAME')
    r4_6 = tf.add_n([r4_5, conv4_63])

    # conv5_x

    # conv5_1
    # conv5_11, 1x1, 512
    conv5_11 = block('conv5_11', r4_6, 512, 1, 2, 'VALID')
    # conv5_12, 3x3, 512
    conv5_12 = block('conv5_12', conv5_11, 512, 3, 1, 'SAME')
    # conv5_13, 1x1, 2048
    conv5_13 = block('conv5_13', conv5_12, 2048, 1, 1, 'VALID')
    r5_1 = tf.add_n([conv2d('pj5', r4_6, 2048, 1, 2, 'VALID'), conv5_13])

    # conv5_2
    # conv5_21, 1x1, 512
    conv5_21 = block('conv5_21', r5_1, 512, 1, 1, 'VALID')
    # conv5_22, 3x3, 512
    conv5_22 = block('conv5_22', conv5_21, 512, 3, 1, 'SAME')
    # conv5_23, 1x1, 2048
    conv5_23 = block('conv5_23', conv5_22, 2048, 1, 1, 'VALID')
    r5_2 = tf.add_n([r5_1, conv5_23])

    # conv5_3
    # conv5_31, 1x1, 512
    conv5_31 = block('conv5_31', r5_2, 512, 1, 1, 'VALID')
    # conv5_32, 3x3, 512
    conv5_32 = block('conv5_32', conv5_31, 512, 3, 1, 'SAME')
    # conv5_33, 1x1, 2048
    conv5_33 = block('conv5_33', conv5_32, 2048, 1, 1, 'VALID')
    r5_3 = tf.add_n([r5_2, conv5_33])

    avgpool6 = avgpool2d('avgpool6', r5_3, 7, 1, 'VALID')
    dense6 = tf.layers.dense(tf.layers.flatten(avgpool6), 12)

    return dense6