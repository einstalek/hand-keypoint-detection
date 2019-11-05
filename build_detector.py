from keras.layers import Activation, BatchNormalization, Conv2D, Concatenate, DepthwiseConv2D, Reshape, Lambda, \
    Convolution2D
from keras import backend as K, Input
from keras.models import Model
import numpy as np

from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization

tf = K.tf
mobilenet = True
separable_filter = False
conv_model = False
dropout_rate = 0.55
W_regularizer = None
init_ = 'glorot_uniform'
conv_has_bias = True  # False for BN
fc_has_bias = True


# Depthwise convolution

def relu6(x):
    return K.relu(x, max_value=6)


def _depthwise_conv_block_detection(input, layer_name, strides=(1, 1),
                                    kernel_size=3,
                                    pointwise_conv_filters=32, alpha=1.0, depth_multiplier=1,
                                    padding='valid',
                                    data_format=None,
                                    activation=None, use_bias=True,
                                    depthwise_initializer='glorot_uniform',
                                    pointwise_initializer='glorot_uniform', bias_initializer="zeros",
                                    bias_regularizer=None, activity_regularizer=None,
                                    depthwise_constraint=None, pointwise_constraint=None,
                                    bias_constraint=None, batch_size=None,
                                    block_id=1, trainable=None, weights=None):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding=padding, depth_multiplier=depth_multiplier, strides=strides,
                        use_bias=False, name=layer_name + '_conv_dw_%d' % block_id)(input)
    x = BatchNormalization(axis=channel_axis, name=layer_name + '_conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name=layer_name + '_conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               # padding='same',
               padding=padding, use_bias=False, strides=(1, 1), name=layer_name + '_conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name=layer_name + '_conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name=layer_name + '_conv_pw_%d_relu' % block_id)(x)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def bn_conv(input_layer, layer_name, nb_filter, nb_row, nb_col, subsample=(1, 1), border_mode='same',
            bias=conv_has_bias):
    tmp_layer = input_layer
    tmp_layer = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, activation=None, border_mode=border_mode,
                              name=layer_name, bias=bias, init=init_, W_regularizer=W_regularizer)(tmp_layer)
    tmp_layer = BatchNormalization(name=layer_name + '_bn')(tmp_layer)
    tmp_layer = Lambda(lambda x: tf.nn.relu(x), name=layer_name + '_nonlin')(tmp_layer)
    return tmp_layer


def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                                         depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3), padding='same',  depth_multiplier=depth_multiplier, strides=strides, use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


########################################################################

def mn_model(image_size,
             n_classes,
             min_scale=0.1,
             max_scale=0.9,
             scales=None,
             aspect_ratios_global=[0.5, 1.0, 2.0],
             aspect_ratios_per_layer=None,
             two_boxes_for_ar1=True,
             limit_boxes=True,
             variances=[1.0, 1.0, 1.0, 1.0],
             coords='centroids',
             normalize_coords=False):
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300

    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one "
                         "needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {},"
                             " but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers,
                                                                               len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, "
                             "but len(scales) == {}.".format(n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed,
        # compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios_conv4_3 = aspect_ratios_per_layer[0]
        aspect_ratios_fc7 = aspect_ratios_per_layer[1]
        aspect_ratios_conv6_2 = aspect_ratios_per_layer[2]
        aspect_ratios_conv7_2 = aspect_ratios_per_layer[3]
        aspect_ratios_conv8_2 = aspect_ratios_per_layer[4]
        aspect_ratios_conv9_2 = aspect_ratios_per_layer[5]
    else:
        aspect_ratios_conv4_3 = aspect_ratios_global
        aspect_ratios_fc7 = aspect_ratios_global
        aspect_ratios_conv6_2 = aspect_ratios_global
        aspect_ratios_conv7_2 = aspect_ratios_global
        aspect_ratios_conv8_2 = aspect_ratios_global
        aspect_ratios_conv9_2 = aspect_ratios_global

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv4_3 = n_boxes[0]  # 4 boxes per cell for the original implementation
        n_boxes_fc7 = n_boxes[1]  # 6 boxes per cell for the original implementation
        n_boxes_conv6_2 = n_boxes[2]  # 6 boxes per cell for the original implementation
        n_boxes_conv7_2 = n_boxes[3]  # 6 boxes per cell for the original implementation
        n_boxes_conv8_2 = n_boxes[4]  # 4 boxes per cell for the original implementation
        n_boxes_conv9_2 = n_boxes[5]  # 4 boxes per cell for the original implementation
    else:  # If only a global aspect ratio list was passed,
        # then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes_conv4_3 = n_boxes
        n_boxes_fc7 = n_boxes
        n_boxes_conv6_2 = n_boxes
        n_boxes_conv7_2 = n_boxes
        n_boxes_conv8_2 = n_boxes
        n_boxes_conv9_2 = n_boxes

    print("Height, Width, Channels :", image_size[0], image_size[1], image_size[2])
    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    input_shape = (img_height, img_width, img_channels)

    img_input = Input(shape=input_shape)

    alpha = 1.0
    depth_multiplier = 1

    x = Lambda(lambda z: z / 255.,  # Convert input feature range to [-1,1]
               output_shape=(img_height, img_width, img_channels), name='lambda1')(img_input)
    x = Lambda(lambda z: z - 0.5,  # Convert input feature range to [-1,1]
               output_shape=(img_height, img_width, img_channels), name='lambda2')(x)
    x = Lambda(lambda z: z * 2.,  # Convert input feature range to [-1,1]
               output_shape=(img_height, img_width, img_channels), name='lambda3')(x)

    x = _conv_block(x, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block_classification(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=10)
    # 11 conv4_3 (300x300)-> 19x19
    conv4_3 = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block_classification(conv4_3, 1024, alpha, depth_multiplier,
                                             strides=(2, 2), block_id=12)  # (300x300) -> 10x10
    fc7 = _depthwise_conv_block_classification(x, 1024, alpha, depth_multiplier,
                                               block_id=13)  # 13 fc7 (300x300) -> 10x10

    conv6_1 = bn_conv(fc7, 'detection_conv6_1', 256, 1, 1, subsample=(1, 1), border_mode='same', bias=conv_has_bias)
    conv6_2 = _depthwise_conv_block_detection(input=conv6_1, layer_name='detection_conv6_2', strides=(2, 2),
                                              pointwise_conv_filters=512, alpha=alpha,
                                              depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                              block_id=1)

    conv7_1 = bn_conv(conv6_2, 'detection_conv7_1', 128, 1, 1, subsample=(1, 1), border_mode='same', bias=conv_has_bias)
    conv7_2 = _depthwise_conv_block_detection(input=conv7_1, layer_name='detection_conv7_2', strides=(2, 2),
                                              pointwise_conv_filters=256, alpha=alpha,
                                              depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                              block_id=2)
    # conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='detection_conv7_1')(conv6_2)
    # conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='detection_conv7_2')(conv7_1)

    conv8_1 = bn_conv(conv7_2, 'detection_conv8_1', 128, 1, 1, subsample=(1, 1), border_mode='same', bias=conv_has_bias)

    conv8_2 = _depthwise_conv_block_detection(input=conv8_1, layer_name='detection_conv8_2', strides=(2, 2),
                                              pointwise_conv_filters=256, alpha=alpha,
                                              depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                              block_id=3)

    # # conv8_2 = bn_conv(conv8_1, 'detection_conv8_2', 256, 2, 2, subsample =(1,1), border_mode ='same', bias=conv_has_bias)

    conv9_1 = bn_conv(conv8_2, 'detection_conv9_1', 64, 1, 1, subsample=(1, 1), border_mode='same', bias=conv_has_bias)
    # conv9_2 = bn_conv(conv9_1, 'detection_conv9_2', 128, 3, 3, subsample =(2,2), border_mode ='same', bias=conv_has_bias)

    conv9_2 = _depthwise_conv_block_detection(input=conv9_1, layer_name='detection_conv9_2', strides=(2, 2),
                                              pointwise_conv_filters=256, alpha=alpha,
                                              depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                              block_id=4)

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='detection_conv4_3_norm')(conv4_3)

    conv4_3_norm_mbox_conf = _depthwise_conv_block_detection(input=conv4_3_norm,
                                                             layer_name='detection_conv4_3_norm_mbox_conf',
                                                             strides=(1, 1),
                                                             pointwise_conv_filters=n_boxes_conv4_3 * n_classes,
                                                             alpha=alpha, depth_multiplier=depth_multiplier,
                                                             padding='same', use_bias=True, block_id=1)

    fc7_mbox_conf = _depthwise_conv_block_detection(input=fc7, layer_name='detection_fc7_mbox_conf', strides=(1, 1),
                                                    pointwise_conv_filters=n_boxes_fc7 * n_classes, alpha=alpha,
                                                    depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                                    block_id=2)
    conv6_2_mbox_conf = _depthwise_conv_block_detection(input=conv6_2, layer_name='detection_conv6_2_mbox_conf',
                                                        strides=(1, 1),
                                                        pointwise_conv_filters=n_boxes_conv6_2 * n_classes, alpha=alpha,
                                                        depth_multiplier=depth_multiplier, padding='same',
                                                        use_bias=True, block_id=3)

    conv7_2_mbox_conf = _depthwise_conv_block_detection(input=conv7_2, layer_name='detection_conv7_2_mbox_conf',
                                                        strides=(1, 1),
                                                        pointwise_conv_filters=n_boxes_conv7_2 * n_classes, alpha=alpha,
                                                        depth_multiplier=depth_multiplier, padding='same',
                                                        use_bias=True, block_id=4)

    conv8_2_mbox_conf = _depthwise_conv_block_detection(input=conv8_2, layer_name='detection_conv8_2_mbox_conf',
                                                        strides=(1, 1),
                                                        pointwise_conv_filters=n_boxes_conv8_2 * n_classes, alpha=alpha,
                                                        depth_multiplier=depth_multiplier, padding='same',
                                                        use_bias=True, block_id=5)
    conv9_2_mbox_conf = _depthwise_conv_block_detection(input=conv9_2, layer_name='detection_conv9_2_mbox_conf',
                                                        strides=(1, 1),
                                                        pointwise_conv_filters=n_boxes_conv9_2 * n_classes, alpha=alpha,
                                                        depth_multiplier=depth_multiplier, padding='same',
                                                        use_bias=True, block_id=6)

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`

    conv4_3_norm_mbox_loc = _depthwise_conv_block_detection(input=conv4_3_norm,
                                                            layer_name='detection_conv4_3_norm_mbox_loc',
                                                            strides=(1, 1), pointwise_conv_filters=n_boxes_conv4_3 * 4,
                                                            alpha=alpha, depth_multiplier=depth_multiplier,
                                                            padding='same', use_bias=True, block_id=1)

    fc7_mbox_loc = _depthwise_conv_block_detection(input=fc7, layer_name='detection_fc7_mbox_loc', strides=(1, 1),
                                                   pointwise_conv_filters=n_boxes_fc7 * 4, alpha=alpha,
                                                   depth_multiplier=depth_multiplier, padding='same', use_bias=True,
                                                   block_id=2)

    conv6_2_mbox_loc = _depthwise_conv_block_detection(input=conv6_2, layer_name='detection_conv6_2_mbox_loc',
                                                       strides=(1, 1), pointwise_conv_filters=n_boxes_conv6_2 * 4,
                                                       alpha=alpha, depth_multiplier=depth_multiplier,
                                                       padding='same', use_bias=True, block_id=3)

    conv7_2_mbox_loc = _depthwise_conv_block_detection(input=conv7_2, layer_name='detection_conv7_2_mbox_loc',
                                                       strides=(1, 1), pointwise_conv_filters=n_boxes_conv7_2 * 4,
                                                       alpha=alpha, depth_multiplier=depth_multiplier, padding='same',
                                                       use_bias=True, block_id=4)

    conv8_2_mbox_loc = _depthwise_conv_block_detection(input=conv8_2, layer_name='detection_conv8_2_mbox_loc',
                                                       strides=(1, 1), pointwise_conv_filters=n_boxes_conv8_2 * 4,
                                                       alpha=alpha, depth_multiplier=depth_multiplier, padding='same',
                                                       use_bias=True, block_id=5)

    conv9_2_mbox_loc = _depthwise_conv_block_detection(input=conv9_2, layer_name='detection_conv9_2_mbox_loc',
                                                       strides=(1, 1), pointwise_conv_filters=n_boxes_conv9_2 * 4,
                                                       alpha=alpha, depth_multiplier=depth_multiplier, padding='same',
                                                       use_bias=True, block_id=5)
    ### Generate the anchor boxes

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`

    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios_conv4_3, two_boxes_for_ar1=two_boxes_for_ar1,
                                             limit_boxes=limit_boxes, variances=variances, coords=coords,
                                             normalize_coords=normalize_coords,
                                             name='detection_conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios_fc7, two_boxes_for_ar1=two_boxes_for_ar1,
                                    limit_boxes=limit_boxes, variances=variances, coords=coords,
                                    normalize_coords=normalize_coords, name='detection_fc7_mbox_priorbox')(fc7)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios_conv6_2, two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes, variances=variances, coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='detection_conv6_2_mbox_priorbox')(conv6_2)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios_conv7_2, two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes, variances=variances, coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='detection_conv7_2_mbox_priorbox')(conv7_2)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios_conv8_2, two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes, variances=variances, coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='detection_conv8_2_mbox_priorbox')(conv8_2)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios_conv9_2, two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes, variances=variances, coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='detection_conv9_2_mbox_priorbox')(conv9_2)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = \
        Reshape((-1, n_classes), name='detection_conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='detection_conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = \
        Reshape((-1, 4), name='detection_conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='detection_fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='detection_conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = \
        Reshape((-1, 8), name='detection_conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='detection_fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = \
        Reshape((-1, 8), name='detection_conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = \
        Reshape((-1, 8), name='detection_conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = \
        Reshape((-1, 8), name='detection_conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = \
        Reshape((-1, 8), name='detection_conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='detection_mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                                 fc7_mbox_conf_reshape,
                                                                 conv6_2_mbox_conf_reshape,
                                                                 conv7_2_mbox_conf_reshape,
                                                                 conv8_2_mbox_conf_reshape,
                                                                 conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='detection_mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                               fc7_mbox_loc_reshape,
                                                               conv6_2_mbox_loc_reshape,
                                                               conv7_2_mbox_loc_reshape,
                                                               conv8_2_mbox_loc_reshape,
                                                               conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='detection_mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                                         fc7_mbox_priorbox_reshape,
                                                                         conv6_2_mbox_priorbox_reshape,
                                                                         conv7_2_mbox_priorbox_reshape,
                                                                         conv8_2_mbox_priorbox_reshape,
                                                                         conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='detection_mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='detection_predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=img_input, outputs=predictions)
    # model = Model(inputs=img_input, outputs=predictions)

    # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
    # be able to generate the default boxes for the matching process outside of the model during training.
    # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
    # Instead, we'll do it in the batch generator function.
    # The spatial dimensions are the same for the confidence and localization predictors,
    # so we just take those of the conf layers.

    predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                fc7_mbox_conf._keras_shape[1:3],
                                conv6_2_mbox_conf._keras_shape[1:3],
                                conv7_2_mbox_conf._keras_shape[1:3],
                                conv8_2_mbox_conf._keras_shape[1:3],
                                conv9_2_mbox_conf._keras_shape[1:3]])

    model_layer = dict([(layer.name, layer) for layer in model.layers])

    # for key in model_layer:
    #    model_layer[key].trainable = True

    # model = Model(img_input, conv9_2)
    # model_layer = dict([(layer.name, layer) for layer in model.layers])
    # predictor_sizes = 0

    return model, model_layer, img_input, predictor_sizes


#################################################################################################

def iou(boxes1, boxes2, coords='centroids'):
    """
    Compute the intersection-over-union similarity (also known as Jaccard similarity)
    of two axis-aligned 2D rectangular boxes or of multiple axis-aligned 2D rectangular
    boxes contained in two arrays with broadcast-compatible shapes.

    Three common use cases would be to compute the similarities for 1 vs. 1, 1 vs. `n`,
    or `n` vs. `n` boxes. The two arguments are symmetric.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)` or 'minmax' for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.

    Returns:
        A 1D Numpy array of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`.
        0 means there is no overlap between two given boxes, 1 means their coordinates are identical.
    """

    if len(boxes1.shape) > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, "
                         "boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    if coords == 'centroids':
        # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    intersection = np.maximum(0, np.minimum(boxes1[:, 1], boxes2[:, 1]) - np.maximum(boxes1[:, 0], boxes2[:, 0])) * \
                   np.maximum(0, np.minimum(boxes1[:, 3], boxes2[:, 3]) - np.maximum(boxes1[:, 2], boxes2[:, 2]))
    union = (boxes1[:, 1] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 2]) + (boxes2[:, 1] - boxes2[:, 0]) * (
            boxes2[:, 3] - boxes2[:, 2]) - intersection

    return intersection / union


def convert_coordinates(tensor, start_index, conversion='minmax2centroids'):
    """
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    two supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (cx, cy, w, h) - the 'centroids' format

    Note that converting from one of the supported formats to another and back is
    an identity operation up to possible rounding errors for integer tensors.

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids'
            or 'centroids2minmax'. Defaults to 'minmax2centroids'.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    """
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 1]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 2] + tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind]  # Set w
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 2]  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set xmin
        tensor1[..., ind + 1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def _greedy_nms2(predictions, iou_threshold=0.45, coords='minmax'):
    """
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function in `decode_y2()`.
    """
    boxes_left = np.copy(predictions)
    maxima = []  # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0:  # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:, 1])  # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index])  # ...copy that box and...
        maxima.append(maximum_box)  # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0)  # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break  # If there are no boxes left after this step, break. Otherwise...
        # ...compare (IoU) the other left over boxes to the maximum box...
        similarities = iou(boxes_left[:, 2:], maximum_box[2:], coords=coords)
        # ...so that we can remove the ones that overlap too much with the maximum box
        boxes_left = boxes_left[similarities <= iou_threshold]
    return np.array(maxima)


def decode_y2(y_pred,
              confidence_thresh=0.5,
              iou_threshold=0.45,
              top_k='all',
              input_coords='centroids',
              normalize_coords=False,
              img_height=None,
              img_width=None):
    """
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Optionally performs confidence thresholding and greedy non-maximum suppression afte the decoding stage.

    Note that the decoding procedure used here is not the same as the procedure used in the original Caffe implementation.
    The procedure used here assigns every box its highest confidence as the class and then removes all boxes fro which
    the highest confidence is the background class. This results in less work for the subsequent non-maximum suppression,
    because the vast majority of the predictions will be filtered out just by the fact that their highest confidence is
    for the background class. It is much more efficient than the procedure of the original implementation, but the
    results may also differ.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in any positive
            class required for a given box to be considered a positive prediction. A lower value will result
            in better recall, while a higher value will result in better precision. Do not use this parameter with the
            goal to combat the inevitably many duplicates that an SSD will produce, the subsequent non-maximum suppression
            stage will take care of those. Defaults to 0.5.
        iou_threshold (float, optional): `None` or a float in [0,1]. If `None`, no non-maximum suppression will be
            performed. If not `None`, greedy NMS will be performed after the confidence thresholding stage, meaning
            all boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score. Defaults to 0.45.
        top_k (int, optional): 'all' or an integer with number of highest scoring predictions to be kept for each batch item
            after the non-maximum suppression stage. Defaults to 'all', in which case all predictions left after the NMS stage
            will be kept.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax'
            for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
    """

    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError(
            "If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the "
            "image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(
                img_height, img_width))

    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:, :, -14:-8])  # Slice out the four offset predictions plus two elements
    # whereto we'll write the class IDs and confidences in the next step
    y_pred_converted[:, :, 0] = np.argmax(y_pred[:, :, :-12], axis=-1)  # The indices of the highest confidence values
    # in the one-hot class vectors are the class ID
    y_pred_converted[:, :, 1] = np.amax(y_pred[:, :, :-12], axis=-1)  # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor),
        # exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_converted[:, :, [4, 5]] = np.exp(y_pred_converted[:, :, [4, 5]] * y_pred[:, :, [-2, -1]])
        # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_converted[:, :, [4, 5]] *= y_pred[:, :, [-6, -5]]
        # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred),
        # (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_converted[:, :, [2, 3]] *= y_pred[:, :, [-4, -3]] * y_pred[:, :, [-6, -5]]
        # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_converted[:, :, [2, 3]] += y_pred[:, :, [-8, -7]]
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates,
        # where 'size' refers to w or h, respectively
        y_pred_converted[:, :, 2:] *= y_pred[:, :, -4:]
        # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred),
        # delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:, :, [2, 3]] *= np.expand_dims(y_pred[:, :, -7] - y_pred[:, :, -8], axis=-1)
        # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred),
        # delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:, :, [4, 5]] *= np.expand_dims(y_pred[:, :, -5] - y_pred[:, :, -6], axis=-1)
        y_pred_converted[:, :, 2:] += y_pred[:, :, -8:-4]  # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute
    # coordinates, do that
    if normalize_coords:
        y_pred_converted[:, :, 2:4] *= img_width  # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:, :, 4:] *= img_height  # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array
    # containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted:  # For each image in the batch...
        # ...get all boxes that don't belong to the background class,...
        boxes = batch_item[np.nonzero(batch_item[:, 0])]
        # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
        boxes = boxes[boxes[:, 1] >= confidence_thresh]
        if iou_threshold:  # ...if an IoU threshold is set...
            # ...perform NMS on the remaining boxes.
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='minmax')
        if top_k != 'all' and boxes.shape[0] > top_k:  # If we have more than `top_k` results left at this point...
            # ...get the indices of the `top_k` highest-scoring boxes...
            top_k_indices = np.argpartition(boxes[:, 1], kth=boxes.shape[0] - top_k, axis=0)[boxes.shape[0] - top_k:]
            boxes = boxes[top_k_indices]  # ...and keep only those boxes...
        # ...and now that we're done, append the array of final predictions for this batch item to the output list
        y_pred_decoded.append(boxes)
    return y_pred_decoded
