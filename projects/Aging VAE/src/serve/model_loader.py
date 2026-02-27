from pathlib import Path

import numpy as np

from src.model.vae_factory import build_vae_64, build_vae_128
from src.model.latent_math import get_age_direction, orthogonalize_against
from src.data_loading import (
    load_filtered_utkface_paths,
    load_images_by_paths,
)


def parse_model_resolution(filename: str) -> tuple[int, int]:
    """Extract resolution from model filename. Defaults to 64x64 for legacy files."""
    result = (64, 64)
    if "_128x128" in filename:
        result = (128, 128)
    return result


def get_model_resolution(model_name: str) -> tuple[int, int]:
    """Public helper to get resolution for a model name."""
    return parse_model_resolution(model_name)


def get_available_models(models_dir: str = "models") -> list[tuple[str, str]]:
    """
    Return list of (path, display_name) tuples for available weight files.
    Sorted by modification time descending (most recent first).
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    weight_files = list(models_path.glob("*.weights.h5"))
    if not weight_files:
        return []

    # Sort by mtime descending (most recent first)
    sorted_files = sorted(
        weight_files,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [(str(p), p.name) for p in sorted_files]


def load_vae_from_weights(weights_path: str, latent_dim: int = 512):
    """Build VAE matching the resolution in filename and load weights."""
    filename = Path(weights_path).name
    resolution = parse_model_resolution(filename)

    if resolution == (128, 128):
        vae = build_vae_128(latent_dim=latent_dim)
    else:
        vae = build_vae_64(latent_dim=latent_dim)

    vae.load_weights(weights_path)
    return vae


def load_serve_state(dataset_path: str, models_dir: str = "models"):
    """
    Load VAE, age vector, and paths for serve mode.
    Does NOT load all images—only loads images with age 18 or 75 for age vector.
    Returns dict with: vae, vec_age, paths, y_ages, y_genders, model_choices, default_model.
    Raises FileNotFoundError if no models exist.
    """
    models = get_available_models(models_dir)
    if not models:
        raise FileNotFoundError(
            f"No model weights found in '{models_dir}/'. "
            "Train a model first with: python main.py --train"
        )

    default_path, default_name = models[0]
    model_choices = {name: path for path, name in models}

    # Load paths and metadata only (no images)
    paths, y_ages, y_genders = load_filtered_utkface_paths(dataset_path)
    if len(paths) == 0:
        raise ValueError(
            f"No images found in '{dataset_path}'. "
            "Ensure the UTKFace dataset exists and contains valid images."
        )

    # Get resolution for default model
    default_resolution = parse_model_resolution(default_name)

    # Load only images needed for age vector (ages 18 and 75)
    age_vec_indices = [i for i, a in enumerate(y_ages) if a in (18, 75)]
    if not age_vec_indices:
        raise ValueError(
            f"No images with age 18 or 75 in '{dataset_path}'. "
            "get_age_direction requires these anchor ages."
        )
    age_vec_paths = [paths[i] for i in age_vec_indices]
    x_age_vec = load_images_by_paths(age_vec_paths, target_size=default_resolution)
    y_age_vec = y_ages[age_vec_indices]
    y_gender_vec = y_genders[age_vec_indices]

    # Load VAE and compute age vector
    vae = load_vae_from_weights(default_path)
    vec_age = get_age_direction(vae, x_age_vec, y_age_vec, start_age=18, end_age=75)

    # Purify age vector by removing gender component (avoids feminine shift when going younger).
    # Orthogonalization shortens the vector, so we preserve_magnitude to keep the aging
    # effect visually strong without needing to retrain or increase animation intensity.
    z_means, _, _ = vae.encoder.predict(x_age_vec)
    male_mask = y_gender_vec == 0
    female_mask = y_gender_vec == 1
    if np.any(male_mask) and np.any(female_mask):
        mu_male = np.mean(z_means[male_mask], axis=0)
        mu_female = np.mean(z_means[female_mask], axis=0)
        vec_gender = mu_female - mu_male
        vec_age = orthogonalize_against(vec_age, vec_gender, preserve_magnitude=True)

    return {
        "vae": vae,
        "vec_age": vec_age,
        "paths": paths,
        "y_ages": y_ages,
        "y_genders": y_genders,  # Used only for orthogonalization, not exposed to UI
        "model_choices": model_choices,
        "default_model": default_name,
        "default_resolution": default_resolution,
        "models_dir": models_dir,
        "dataset_path": dataset_path,
    }
