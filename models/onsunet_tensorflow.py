# ResNet-50 implementation in TensorFlow

import tensorflow as tf

class OnsuNetTensorFlow:
    def build_model(self, num_classes=1000):
        # input
        i = tf.keras.layers.Input(shape=(48, 48, 1))

        # block 1
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(i)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # block 2
        x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # block 3
        x = tf.keras.layers.Conv2D(filters=192, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # block 4
        x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # block 5
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # output
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(i, x)

    def compile_model(self, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']):
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