# ResNet-50 implementation in TensorFlow

import tensorflow as tf

class ResnetTensorFlow:
    def identity_block(self, input, filter_size, filters):
        f1, f2, f3 = filters
        bn_axis = 3

        # first block
        x = tf.keras.layers.Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
        x = tf.keras.layers.Activation('relu')(x)

        # second block
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

    def convolutional_block(self, input, filter_size, filters, strides=(2, 2)):
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

    def build_model(self, input_shape=(224, 224, 3), num_classes=7):
        # get shape of input
        i = tf.keras.layers.Input(input_shape)

        # zero padding the input with a pad of (3, 3)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(i)

        # stage 1
        x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        # stage 2 (layer 1, 3 blocks)
        x = self.convolutional_block(input=x, filter_size=3, filters=[64, 64, 256], strides=(1, 1))
        x = self.identity_block(input=x, filter_size=3, filters=[64, 64, 256])
        x = self.identity_block(input=x, filter_size=3, filters=[64, 64, 256])

        # stage 3 (layer 2, 4 blocks)
        x = self.convolutional_block(input=x, filter_size=3, filters=[128, 128, 512], strides=(2, 2))
        x = self.identity_block(input=x, filter_size=3, filters=[128, 128, 512])
        x = self.identity_block(input=x, filter_size=3, filters=[128, 128, 512])
        x = self.identity_block(input=x, filter_size=3, filters=[128, 128, 512])

        # stage 4 (layer 3, 6 blocks)
        x = self.convolutional_block(input=x, filter_size=3, filters=[256, 256, 1024], strides=(2, 2))
        x = self.identity_block(input=x, filter_size=3, filters=[256, 256, 1024])
        x = self.identity_block(input=x, filter_size=3, filters=[256, 256, 1024])
        x = self.identity_block(input=x, filter_size=3, filters=[256, 256, 1024])
        x = self.identity_block(input=x, filter_size=3, filters=[256, 256, 1024])
        x = self.identity_block(input=x, filter_size=3, filters=[256, 256, 1024])

        #stage 5 (layer 4, 3 blocks)
        x = self.convolutional_block(input=x, filter_size=3, filters=[512, 512, 2048], strides=(2, 2))
        x = self.identity_block(input=x, filter_size=3, filters=[512, 512, 2048])
        x = self.identity_block(input=x, filter_size=3, filters=[512, 512, 2048])

        # average pooling
        x = tf.keras.layers.AveragePooling2D((2, 2))(x)

        # output layer
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(i, x)

    def compile_model(self, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train_model(self, training_data, validation_data, epochs, steps_per_epoch, validation_steps):
        self.model.fit(
            training_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            validation_steps=validation_steps
        )