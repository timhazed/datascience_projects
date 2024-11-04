import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base_cnn_model import BaseCNNModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleRoofCNNModel(BaseCNNModel):
    def __init__(self, max_pointcloud_length, max_planes, max_perimeter_points, augment=False, augmentation_count=2):
        super().__init__(max_pointcloud_length, max_planes, max_perimeter_points, augment, augmentation_count)
        self.image_shape = (224,224,3)
        self.model = self._build_model()

    def _build_model(self):
        # Image input branch using ResNet50 (pretrained)
        image_input = layers.Input(shape=self.image_shape, dtype=tf.float32, name='image_input')
        base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=self.image_shape, weights='imagenet')

        # Aerial image passed through ResNet50
        base_output = base_model(image_input) 

        # Pooling to condense image features
        image_x = layers.GlobalAveragePooling2D()(base_output)  

        # Point cloud input branch
        pointcloud_input = layers.Input(shape=(self.max_pointcloud_length, 3), dtype=tf.float32, name='pointcloud_input')
        pointcloud_x = layers.Conv1D(128, kernel_size=1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.15))(pointcloud_input)
        pointcloud_x = layers.BatchNormalization()(pointcloud_x)
        pointcloud_x = layers.Conv1D(256, kernel_size=1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.15))(pointcloud_x)
        pointcloud_x = layers.BatchNormalization()(pointcloud_x)
        pointcloud_x = layers.GlobalAveragePooling1D()(pointcloud_x)  # Pooling to condense point cloud features

        # Combine features from both branches
        combined = layers.Concatenate()([image_x, pointcloud_x])
        
        # Dense layer to process the combined features
        combined_x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(combined)
        combined_x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(combined_x)
        
        # Output layers, modified for multi-plane support
        azimuth_output = layers.Dense(self.max_planes, name='azimuth', 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.05))(combined_x)
        height_output = layers.Dense(self.max_planes, name='height', 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.05))(combined_x)
        tilt_output = layers.Dense(self.max_planes, name='tilt', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.03))(combined_x)
        
        # Dense layer for perimeter with a unique temporary name
        perimeter_dense = layers.Dense(self.max_planes * self.max_perimeter_points * 2, name='perimeter_dense', 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.1))(combined_x)
        
        # Reshape perimeter output to the correct dimensions and assign the final name 'perimeter'
        perimeter_output = layers.Reshape((self.max_planes, self.max_perimeter_points, 2), name='perimeter')(perimeter_dense)

        # Define the final model with both inputs and four outputs
        model = tf.keras.Model(inputs=[image_input, pointcloud_input], 
                            outputs=[azimuth_output, height_output, tilt_output, perimeter_output])
        return model


    def train(self, images, pointclouds, labels, checkpoint_dir='checkpoints'):
        # Unpack azimuths, tilts, heights, perimeters
        azimuth_labels = labels[0]
        tilt_labels = labels[1]
        height_labels = labels[2]
        perimeter_labels = labels[3]

        # Initial split: 70% train, 30% remaining (to be split into validation and test)
        images_train, images_remaining, pointclouds_train, pointclouds_remaining, azimuth_train, azimuth_remaining, \
        height_train, height_remaining, tilt_train, tilt_remaining, perimeter_train, perimeter_remaining = train_test_split(
            np.array(images), np.array(pointclouds),
            np.array(azimuth_labels), np.array(height_labels),
            np.array(tilt_labels), np.array(perimeter_labels),
            test_size=0.3, random_state=42
        )

        # Second split on the remaining 30%: 20% validation and 10% test
        images_val, images_test, pointclouds_val, pointclouds_test, azimuth_val, azimuth_test, \
        height_val, height_test, tilt_val, tilt_test, perimeter_val, perimeter_test = train_test_split(
            images_remaining, pointclouds_remaining,
            azimuth_remaining, height_remaining,
            tilt_remaining, perimeter_remaining,
            test_size=1/3, random_state=42  # 1/3 of the remaining 30% to get 10% for test
        )

        # Logging shapes to verify consistent data structure
        logging.info(f"Images train shape: {images_train.shape}")
        logging.info(f"Pointclouds train shape: {pointclouds_train.shape}")
        logging.info(f"Azimuth train shape: {azimuth_train.shape}")
        logging.info(f"Height train shape: {height_train.shape}")
        logging.info(f"Tilt train shape: {tilt_train.shape}")
        logging.info(f"Perimeter train shape: {perimeter_train.shape}")

        # If augmentation is enabled, augment the training data
        if self.augment:
            images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train = self._augment_training_data(
                images_train, pointclouds_train, azimuth_train, height_train, tilt_train, perimeter_train, self.augment_count)
            
        loss_weights = {'azimuth': 0.2, 'height': 1.5, 'tilt': 1.5, 'perimeter': 0.3}

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # Compile the model with multiple outputs
        self.model.compile(optimizer=optimizer, 
                           loss={'azimuth': 'mse', 'height': 'mse', 'tilt': 'mse', 'perimeter': 'mse'}, 
                           loss_weights=loss_weights)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define the checkpoint callback to save models during training
        checkpoint_path = os.path.join(checkpoint_dir, 'simple_roof_cnn_epoch_{epoch:02d}_loss_{loss:.4f}.h5')
        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            monitor='val_loss',  # Track validation loss for model checkpointing
            verbose=1, 
            save_best_only=True,  # Save the best models based on validation loss
            mode='auto'
        )

        # Define callbacks: early stopping and learning rate reduction
        early_stopping_general = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        early_stopping_tilt = tf.keras.callbacks.EarlyStopping(monitor='val_tilt_loss', patience=8, restore_best_weights=True, verbose=1)
        early_stopping_perimeter = tf.keras.callbacks.EarlyStopping(monitor='val_perimeter_loss', patience=8, restore_best_weights=True, verbose=1)

        reduce_lr_tilt = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_tilt_loss', factor=0.2, patience=3, min_lr=1e-4, verbose=1)
        reduce_lr_perimeter = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_perimeter_loss', factor=0.2, patience=3, min_lr=1e-4, verbose=1)
        
        logging.info(f"Model Summary: {self.model.summary()}")
        # Proceed with the rest of the training setup if shapes are consistent
        history = self.model.fit(
            [images_train, pointclouds_train],
            {'azimuth': azimuth_train, 'height': height_train, 'tilt': tilt_train, 'perimeter': perimeter_train},
            validation_data=(
                [images_val, pointclouds_val],
                {'azimuth': azimuth_val, 'height': height_val, 'tilt': tilt_val, 'perimeter': perimeter_val}
            ),
            epochs=80,
            batch_size=32,
            callbacks=[early_stopping_tilt, early_stopping_general,early_stopping_perimeter,reduce_lr_perimeter, reduce_lr_tilt]
        )

        return history


