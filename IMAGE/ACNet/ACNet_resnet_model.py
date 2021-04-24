from tensorflow.keras import layers, Model, Sequential
import  tensorflow as tf

class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2", activation=tf.identity)
        self.gap = layers.GlobalAvgPool2D(name='gap')
        self.fc1 = layers.Dense(out_channel, activation='relu')
        self.fc2 = layers.Dense(out_channel, activation=tf.identity)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv_c1 = layers.Conv2D(out_channel, kernel_size=1, strides=1, activation='relu', name='conv_c1')
        self.conv_c2 = layers.Conv2D(out_channel, kernel_size=1, strides=1, activation=tf.identity, name='conv_c2')
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()
        self.mul = layers.Multiply()
        self.ch_out = out_channel

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x_3 = self.conv2(x)

        shape = x_3.get_shape().as_list()
        x_gap = self.gap(x)
        x_gap = self.fc1(x_gap)
        x_gap = self.fc2(x_gap)
        x_gap = tf.reshape(x_gap, [-1, self.ch_out, 1, 1])
        x_gap = tf.tile(x_gap, [1, 1, shape[2], shape[3]])

        x_concat = tf.concat([x_3, x_gap], axis=1)
        x_concat = self.conv_c1(x_concat)
        x_concat = self.conv_c2(x_concat)
        x_concat = tf.sigmoid(x_concat)

        x = x_3 + x_gap * x_concat
        x = self.bn2(x, training=training)
        x= self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 256, 256, 1)
    input_image = layers.Input(shape=(im_height, im_width, 1), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def ACNet_resnet34(im_width=256, im_height=256, num_classes=9, include_top=True):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def ACNet_resnet50(im_width=256, im_height=256, num_classes=9, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def ACNet_resnet101(im_width=256, im_height=256, num_classes=9, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)