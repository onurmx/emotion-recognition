import tensorflow as tf

class ResNetTensorFlow:
    def identity_block(input, filter_size, filters):
        f1, f2, f3 = filters

        bn_axis = 3

        # first block
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation('relu')(x)

        # second block (bottleneck but size remains same due to padding)
        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(filter_size, filter_size), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation('relu')(x)

        # third block
        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)

        # add the input to third block
        x = tf.keras.layers.Add()([x, input])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def convolutional_block(input, filter_size, filters, strides=(2, 2)):
        f1, f2, f3 = filters

        bn_axis = 3

        # first block
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding='valid')(input)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation('relu')(x)

        # second block
        x = tf.keras.layers.Conv2D(filters=f2, kernel_size=(filter_size, filter_size), strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation('relu')(x)

        # third block
        x = tf.keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)

        # apply convolution to shortcut to get matching dimensions to third block's output dimensions
        shortcut = tf.keras.layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=strides, padding='valid')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)

        # add the shortcut to third block
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        return x

    def ResNet50(self, input_shape=(224, 224, 3), classes=6):
        # get shape of input
        i = tf.keras.layers.Input(input_shape)

        # zero padding the input with a pad of (3, 3)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(i)

        # stage 1
        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # stage 2
        x = self.convolutional_block(x, f=3, filters=[64, 64, 256], s=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256])
        x = self.identity_block(x, 3, [64, 64, 256])

        # stage 3
        x = self.convolutional_block(x, f=3, filters=[128, 128, 512], s=(2, 2))
        x = self.identity_block(x, 3, [128, 128, 512])
        x = self.identity_block(x, 3, [128, 128, 512])
        x = self.identity_block(x, 3, [128, 128, 512])

        # stage 4
        x = self.convolutional_block(x, f=3, filters=[256, 256, 1024], s=(2, 2))
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])
        x = self.identity_block(x, 3, [256, 256, 1024])

        #stage 5
        x = self.convolutional_block(x, f=3, filters=[512, 512, 2048], s=(2, 2))
        x = self.identity_block(x, 3, [512, 512, 2048])
        x = self.identity_block(x, 3, [512, 512, 2048])

        # average pooling
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        # output layer
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(classes, activation='softmax')(x)

        model = tf.keras.Model(i, x)
        return model

    def compile_model(model):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(model, train_data, train_labels, test_data, test_labels, epochs=10):
        model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels))
        return model