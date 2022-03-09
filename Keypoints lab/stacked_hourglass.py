import tensorflow as tf

from tensorflow.keras import layers as tf_layers


def residual_block(x, n_filters, mode='residual'):
    if mode == 'residual':
        skip = tf_layers.Conv2D(n_filters, (1, 1), activation='relu')(x)

        x = tf_layers.Conv2D(n_filters // 2, (1, 1), activation='relu')(x)
        x = tf_layers.Conv2D(n_filters // 2, (3, 3), padding='same', activation='relu')(x)
        x = tf_layers.Conv2D(n_filters, (1, 1), activation='relu')(x)
        x = tf_layers.Add()([skip, x])

    elif mode == 'simple':
        x = tf_layers.Conv2D(n_filters, (3, 3), padding='same', activation='relu')(x)

    return x


def hourglass_block(x, n_filters, max_filters, mode='residual'):
    if n_filters >= max_filters:
        x = residual_block(x, n_filters, mode)
        x = residual_block(x, n_filters // 2, mode)
    else:
        x = residual_block(x, n_filters, mode)
        skip = x
        x = tf_layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = hourglass_block(x, n_filters * 2, max_filters, mode)

        x = tf_layers.UpSampling2D((2, 2))(x)
        x = tf_layers.Add()([skip, x])
        x = residual_block(x, n_filters // 2, mode)

    return x


def create_stacked_hourglass_model(img_input, n_keypoints, n_hourglasses, start_filters, max_filters, mode='residual'):
    """
    create a tensorflow stacked hourglass model as described here: https://arxiv.org/abs/1603.06937
    :param img_input: input for the model
    :param n_keypoints: number of keypoints
    :param n_hourglasses: number of hourglasses stacked together
    :param start_filters: the smallest number of conv filters used (top & bottom of hourglasses)
    :param max_filters: the biggest number of conv filters used (bottleneck). While traversing each hourglass,
    the filter count is repeatedly multiplied by 2
    :param mode: if set to 'residual', each hourglass step is a residual module; if set to 'simple', each hourglass is
    a basic Conv2D
    :return: a tensorflow model
    """
    x = tf_layers.Conv2D(start_filters, (3, 3), padding='same', activation='relu')(
        img_input)  # transform to the hourglass base size

    skip = x
    output_list = []
    for i in range(n_hourglasses):
        x = hourglass_block(x, start_filters * 2, max_filters, mode)

        output = tf_layers.Conv2D(n_keypoints, (1, 1), activation='linear', name=f'output_{i}')(x)  # produce heatmaps
        output_list.append(output)
        mapped_output = tf_layers.Conv2D(start_filters, (1, 1), activation='relu')(
            output)  # map back to the base size

        x = tf_layers.Add()([skip, mapped_output, x])
        skip = x

    stacked_hourglass_model = tf.keras.Model(img_input, output_list)
    return stacked_hourglass_model


if __name__ == '__main__':
    img_input = tf.keras.Input(shape=(128, 96, 3))
    model = create_stacked_hourglass_model(img_input, 9, 4, 32, 256, mode='simple')
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error)

    sample_inputs = tf.zeros([8, 128, 96, 3])
    sample_outputs = tf.ones([8, 128, 96, 9])
    model.fit(sample_inputs, sample_outputs, epochs=10)
