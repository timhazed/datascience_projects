import argparse
import datetime
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.model.vae_factory import build_vae_64, build_vae_128
from src.model.latent_math import get_age_direction, orthogonalize_against
from src.data_loading import load_filtered_utkface
from src.serve import load_serve_state, create_app

def setup_gpu():
    # Check for Apple Silicon GPU (MPS)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Active GPU: {gpu_devices}")
        # This allows TensorFlow to use the GPU memory efficiently
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print("GPU not found. Ensure tensorflow-metal is installed.")


def visualize_age_progression(vae, original_img, age_vec, steps=9):
    """
    Visualizes a row of faces by moving along the age vector (young to old).
    """
    z_base, _, _ = vae.encoder.predict(np.expand_dims(original_img, 0), verbose=0)
    age_range_width = 4.0

    plt.figure(figsize=(15, 3))
    for j in range(steps):
        alpha_a = ((j / (steps - 1)) - 0.5) * age_range_width
        z_modified = z_base + (alpha_a * age_vec)
        gen_img = vae.decoder.predict(z_modified, verbose=0)[0]
        plt.subplot(1, steps, j + 1)
        plt.imshow(gen_img)
        plt.axis("off")
        plt.title(f"Age: {alpha_a:.1f}")
    plt.tight_layout()
    plt.show()

def plot_vae_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['reconstruction_loss'], label='Reconstruction')
    plt.plot(history.history['total_loss'], label='Total')
    plt.title('Reconstruction & Total Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['kl_loss'], label='KL Loss', color='orange')
    plt.title('KL Divergence (Latent Structure)')
    plt.legend()
    
    plt.show()

def run_training(dataset_path, resolution=64):
    # Execution on your M4 Max
    target_size = (resolution, resolution)
    x_train, y_ages, y_genders = load_filtered_utkface(dataset_path, target_size=target_size)
    print(f"Final dataset size: {len(x_train)} samples at {resolution}x{resolution}")

    # 1. Initialize and Train
    if resolution == 128:
        vae = build_vae_128(latent_dim=512)
    else:
        vae = build_vae_64(latent_dim=512)

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, f"vae_weights_{timestamp}_{resolution}x{resolution}.weights.h5")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='reconstruction_loss',
            mode='min',
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            # Change extension to .weights.h5
            filepath=checkpoint_path, 
            monitor='total_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True 
        ),
    ]
    print(f"Training VAE... Saving checkpoints to {checkpoint_path}")

    # Train for 200 epochs
    history = vae.fit(x_train, x_train, epochs=200, batch_size=256, callbacks=callbacks)
    plot_vae_history(history)

    # 1. Calculate age vector (18 and 75 as anchors)
    vec_age = get_age_direction(vae, x_train, y_ages, start_age=18, end_age=75)

    # 2. Purify age vector by removing gender component (avoids feminine shift when going younger).
    # preserve_magnitude=True rescales the purified vector to match the original length,
    # so the age progression visualization remains visually strong.
    z_means, _, _ = vae.encoder.predict(x_train, batch_size=128)
    mu_male = np.mean(z_means[y_genders == 0], axis=0)
    mu_female = np.mean(z_means[y_genders == 1], axis=0)
    vec_gender = mu_female - mu_male
    vec_age = orthogonalize_against(vec_age, vec_gender, preserve_magnitude=True)

    # 3. Visualize age progression
    test_subject = x_train[10]
    visualize_age_progression(vae, test_subject, vec_age)

def run_serve(dataset_path: str, models_dir: str = "models") -> None:
    """Launch the Gradio web interface."""
    try:
        serve_state = load_serve_state(dataset_path, models_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    demo = create_app(serve_state)
    # show_api=False avoids gradio-client bug with boolean JSON schema (additionalProperties: false)
    demo.launch(show_api=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aging VAE: Train or serve the face aging model."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/UTKFace",
        help="Path to UTKFace dataset (default: data/UTKFace)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[64, 128],
        default=64,
        help="Image resolution for training: 64 or 128 (default: 64)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train the VAE model")
    mode.add_argument("--serve", action="store_true", help="Launch Gradio web interface")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_gpu()

    if args.train:
        run_training(args.dataset_path, resolution=args.resolution)
    else:
        run_serve(args.dataset_path)