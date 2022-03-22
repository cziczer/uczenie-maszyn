import tensorflow as tf

from tensorflow.keras import layers as tf_layers


def residual_block(x, n_filters, mode='residual', apply_skip_conv=False):
    if mode == 'residual':
        skip = tf_layers.Conv2D(n_filters, (1, 1))(x)
        skip = tf_layers.BatchNormalization()(skip)
        skip = tf_layers.LeakyReLU(alpha=0.1)(skip)

        x = tf_layers.Conv2D(n_filters // 2, (1, 1))(x)
        x = tf_layers.BatchNormalization()(x)
        x = tf_layers.LeakyReLU(alpha=0.1)(x)

        x = tf_layers.Conv2D(n_filters // 2, (3, 3), padding='same')(x)
        x = tf_layers.BatchNormalization()(x)
        x = tf_layers.LeakyReLU(alpha=0.1)(x)

        x = tf_layers.Conv2D(n_filters, (1, 1))(x)
        x = tf_layers.BatchNormalization()(x)
        x = tf_layers.LeakyReLU(alpha=0.1)(x)

        x = tf_layers.Add()([skip, x])
        # x = tf_layers.Dropout(0.2)(x)

    elif mode == 'simple':
        x = tf_layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu')(x)

    return x


def hourglass_block(x, current_shape, n_filters=256, smallest_shape=4, mode='residual'):
    if current_shape <= smallest_shape:
        x = residual_block(x, n_filters, mode)
    else:
        x = residual_block(x, n_filters, mode)
        skip = x
        x = tf_layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = hourglass_block(x, current_shape // 2, n_filters, smallest_shape, mode)

        x = tf_layers.UpSampling2D((2, 2))(x)
        x = tf_layers.Add()([skip, x])
        x = residual_block(x, n_filters, mode)

    return x


def create_stacked_hourglass_model(img_input, n_keypoints, n_hourglasses, n_filters=256, smallest_shape=4,
                                   mode='residual'):
    """
    create a tensorflow stacked hourglass model as described here: https://arxiv.org/abs/1603.06937
    :param smallest_shape: bottleneck shape of each hourglass
    :param n_filters: number of Conv2D filters used
    :param img_input: input for the model
    :param n_keypoints: number of keypoints
    :param n_hourglasses: number of hourglasses stacked together
    :param mode: if set to 'residual', each hourglass step is a residual module; if set to 'simple', each hourglass is
    a basic Conv2D
    :return: a tensorflow model
    """
    # x = tf_layers.Conv2D(n_filters, (7, 7), strides=(1, 1), padding='same', activation='relu')(img_input)
    # x = tf_layers.BatchNormalization()(x)
    # x = tf_layers.LeakyReLU(alpha=0.1)(x)

    x = residual_block(img_input, n_filters, mode)
    # x = tf_layers.Dropout(0.2)(x)
    x = tf_layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    skip = x
    output_list = []
    for i in range(n_hourglasses):
        x = hourglass_block(x, img_input.shape[1], n_filters, smallest_shape, mode)

        # produce heatmaps
        output = tf_layers.UpSampling2D((2, 2))(x)
        output = tf_layers.Conv2D(n_keypoints, (1, 1), activation='sigmoid', name=f'output_{i}')(output)
        output_list.append(output)

        # map back to the base size
        mapped_output = tf_layers.Conv2D(n_filters, (1, 1), activation='relu')(output)
        mapped_output = tf_layers.MaxPooling2D((2, 2), strides=(2, 2))(mapped_output)

        x = tf_layers.Add()([skip, mapped_output, x])
        skip = x

    stacked_hourglass_model = tf.keras.Model(img_input, output_list)
    return stacked_hourglass_model


if __name__ == '__main__':
    img_input = tf.keras.Input(shape=(128, 128, 3))
    model = create_stacked_hourglass_model(img_input, 9, 4, 256, 16, mode='simple')
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)

    sample_inputs = tf.zeros([8, 128, 128, 3])
    sample_outputs = tf.ones([8, 128, 128, 9])
    model.fit(sample_inputs, sample_outputs, epochs=10)
