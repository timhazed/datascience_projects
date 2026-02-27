# Multi-Resolution VAE Support Plan

## Overview

This document outlines the changes needed to support both 64x64 (low-res) and 128x128 (high-res) VAE models within the same application, with automatic detection based on model filename conventions.

## Current State

| Component | Current Behavior |
|-----------|-----------------|
| `vae_factory.py` | `INPUT_SHAPE = (64, 64, 3)` hardcoded; `build_vae_high_res` is a deeper network but still 64x64 |
| `loader.py` | `target_size=(64, 64)` default in all functions |
| `model_loader.py` | Always calls `build_vae_high_res(latent_dim=512)` |
| `main.py` | Saves weights as `vae_weights_{timestamp}.weights.h5` |
| `gradio_app.py` | Loads images via `load_single_image(path)` which uses 64x64 |

## Proposed Naming Convention

Model weight files will encode resolution in the filename:

```
vae_weights_{timestamp}_{resolution}.weights.h5
```

Examples:
- `vae_weights_20260226-1603_64x64.weights.h5`
- `vae_weights_20260226-1710_128x128.weights.h5`

### Resolution Parsing Logic

```python
def parse_model_resolution(filename: str) -> tuple[int, int]:
    """Extract resolution from model filename. Defaults to 64x64 for legacy files."""
    if "_128x128" in filename:
        return (128, 128)
    elif "_64x64" in filename:
        return (64, 64)
    else:
        # Legacy files without resolution tag assumed to be 64x64
        return (64, 64)
```

## Implementation Steps

### 1. Update `vae_factory.py`

Add a true 128x128 factory function. The architecture must account for the larger spatial dimensions.

```python
INPUT_SHAPE_64 = (64, 64, 3)
INPUT_SHAPE_128 = (128, 128, 3)

def build_vae_64(latent_dim=512):
    """VAE for 64x64 images (current build_vae_high_res)."""
    # ... existing build_vae_high_res code with INPUT_SHAPE_64 ...

def build_vae_128(latent_dim=512):
    """VAE for 128x128 images."""
    # --- ENCODER ---
    encoder_inputs = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)  # Extra level for 128px

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
```

**Exports to add:**
```python
__all__ = ["build_vae", "build_vae_64", "build_vae_128", "INPUT_SHAPE_64", "INPUT_SHAPE_128"]
```

### 2. Update `model_loader.py`

Add resolution detection and factory selection.

```python
from src.model.vae_factory import build_vae_64, build_vae_128

def parse_model_resolution(filename: str) -> tuple[int, int]:
    """Extract resolution from model filename."""
    if "_128x128" in filename:
        return (128, 128)
    return (64, 64)  # default for legacy and 64x64 files

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

def get_model_resolution(model_name: str) -> tuple[int, int]:
    """Public helper for gradio_app to get resolution for image loading."""
    return parse_model_resolution(model_name)
```

**Update `load_serve_state`:**
- Store resolution in returned dict
- Pass resolution to `load_images_by_paths` for age vector computation

### 3. Update `loader.py`

No changes needed to function signatures - `target_size` parameter already exists. Callers will pass the appropriate size.

### 4. Update `gradio_app.py`

Modify to use model-specific resolution when loading images.

```python
from src.serve.model_loader import get_model_resolution

def get_vae_and_vec(model_name: str):
    if model_name not in model_cache:
        path = model_choices[model_name]
        resolution = get_model_resolution(model_name)  # NEW
        vae = load_vae_from_weights(path)

        # Load age vector images at correct resolution
        age_vec_indices = [i for i, a in enumerate(y_ages) if a in (18, 75)]
        age_vec_paths = [paths[i] for i in age_vec_indices]
        x_age = load_images_by_paths(age_vec_paths, target_size=resolution)  # CHANGED
        y_age = y_ages[age_vec_indices]
        vec_age = get_age_direction(vae, x_age, y_age, start_age=18, end_age=75)
        model_cache[model_name] = (vae, vec_age, resolution)  # Store resolution
    return model_cache[model_name]

def on_random(model_name):
    path_idx = np.random.randint(0, len(paths))
    path = paths[path_idx]
    vae, vec_age, resolution = get_vae_and_vec(model_name)  # Unpack resolution
    img = load_single_image(path, target_size=resolution)   # Use it
    output = age_person(vae, img, vec_age, intensity=0.0)
    return _img_for_display(output), img
```

### 5. Update `main.py` Training

Modify checkpoint naming and add CLI argument for resolution.

```python
parser.add_argument(
    "--resolution",
    type=int,
    choices=[64, 128],
    default=64,
    help="Image resolution for training (64 or 128)",
)

def run_training(dataset_path, resolution=64):
    target_size = (resolution, resolution)
    x_train, y_ages, y_genders = load_filtered_utkface(dataset_path, target_size=target_size)

    if resolution == 128:
        vae = build_vae_128(latent_dim=512)
    else:
        vae = build_vae_64(latent_dim=512)

    # Include resolution in checkpoint filename
    checkpoint_path = os.path.join(
        save_dir,
        f"vae_weights_{timestamp}_{resolution}x{resolution}.weights.h5"
    )
    # ... rest of training ...
```

## Migration for Existing Models

Existing model files without resolution tags will be treated as 64x64 (the current default). To make this explicit, rename them:

```bash
# Example migration script
for f in models/vae_weights_*.weights.h5; do
    if [[ ! "$f" =~ _(64x64|128x128)\. ]]; then
        mv "$f" "${f%.weights.h5}_64x64.weights.h5"
    fi
done
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/model/vae_factory.py` | Add `build_vae_64`, `build_vae_128`, rename constants |
| `src/serve/model_loader.py` | Add `parse_model_resolution`, `get_model_resolution`, update `load_vae_from_weights` |
| `src/serve/gradio_app.py` | Pass resolution to image loading functions |
| `src/data_loading/loader.py` | No changes (already parameterized) |
| `main.py` | Add `--resolution` arg, update training and checkpoint naming |

## Testing Checklist

- [ ] Train a 64x64 model, verify filename contains `_64x64`
- [ ] Train a 128x128 model, verify filename contains `_128x128`
- [ ] Load legacy model (no resolution tag), verify treated as 64x64
- [ ] Web app correctly loads 64x64 model and images
- [ ] Web app correctly loads 128x128 model and images
- [ ] Switching models in dropdown loads correct resolution
- [ ] Animation works correctly with both resolutions
