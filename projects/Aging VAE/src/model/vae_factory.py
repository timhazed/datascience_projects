from tensorflow.keras import layers, Model
from src.model.sampling import Sampling
from src.model.vae import VAE

INPUT_SHAPE_64 = (64, 64, 3)
INPUT_SHAPE_128 = (128, 128, 3)


def build_vae_64(latent_dim=512):
    # --- ENCODER ---
    enc_inputs = layers.Input(shape=INPUT_SHAPE_64)
    
    # Block 1: 64x64 -> 32x32
    x = layers.Conv2D(64, 3, strides=2, padding="same")(enc_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Block 2: 32x32 -> 16x16
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Block 3: 16x16 -> 8x8
    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x) # Increased intermediate density
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")

    # --- DECODER ---
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 256))(x)
    
    # Upsample 1: 8x8 -> 16x16
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Upsample 2: 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Upsample 3: 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    return VAE(encoder, decoder)


def build_vae_128(latent_dim=512):
    """VAE for 128x128 images."""
    # --- ENCODER ---
    encoder_inputs = layers.Input(shape=INPUT_SHAPE_128)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder_128")

    # --- DECODER ---
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 256))(x)

    # Upsampling chain: 8->16->32->64->128
    x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)

    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder_128")

    return VAE(encoder, decoder)