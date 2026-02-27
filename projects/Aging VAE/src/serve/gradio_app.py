"""Gradio web interface for the Aging VAE."""

import time
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

from src.model.latent_math import age_person, get_age_direction, orthogonalize_against
from src.serve.model_loader import load_vae_from_weights, get_model_resolution
from src.data_loading import load_single_image, load_images_by_paths

DISPLAY_SIZE = 128
CONTAINER_PADDING = 75  # Extra space so Output buttons don't occlude the image
ANIMATION_FRAME_DELAY = 0.05  # Seconds between frames (increase to slow down)
ANIMATION_STEPS = 50  # Steps for both animate and age slider (0 to MAX_AGE_INTENSITY)
MAX_AGE_INTENSITY = 3.5
DEFAULT_IMAGE_FILENAME = "61_0_0_20170113185438888.jpg.chip.jpg"


def _img_for_display(img: np.ndarray) -> np.ndarray | None:
    """Convert float32 [0,1] to uint8 [0,255] and upscale to fill display area."""
    if img is None:
        return None
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if out.shape[0] < DISPLAY_SIZE or out.shape[1] < DISPLAY_SIZE:
        out = cv2.resize(out, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
    return out


def create_app(serve_state: dict) -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    model_choices = serve_state["model_choices"]
    default_model = serve_state["default_model"]
    paths = serve_state["paths"]
    y_ages = serve_state["y_ages"]
    y_genders = serve_state["y_genders"]

    # Model cache: stores (vae, vec_age, resolution) tuples
    # Derive resolution from default model name (never hardcode—could be 64 or 128)
    default_resolution = get_model_resolution(default_model)
    model_cache = {
        default_model: (serve_state["vae"], serve_state["vec_age"], default_resolution),
    }

    def get_vae_and_vec(model_name: str):
        if model_name not in model_cache:
            path = model_choices[model_name]
            resolution = get_model_resolution(model_name)
            vae = load_vae_from_weights(path)
            # Need images for age vector; load only age 18 and 75
            age_vec_indices = [i for i, a in enumerate(y_ages) if a in (18, 75)]
            age_vec_paths = [paths[i] for i in age_vec_indices]
            x_age = load_images_by_paths(age_vec_paths, target_size=resolution)
            y_age = y_ages[age_vec_indices]
            y_gender_sub = y_genders[age_vec_indices]
            vec_age = get_age_direction(vae, x_age, y_age, start_age=18, end_age=75)
            # Purify age vector by removing gender component. preserve_magnitude=True
            # rescales the result so the aging effect stays strong (orthogonalization
            # shortens the vector; without rescaling the animation would look weak).
            z_means, _, _ = vae.encoder.predict(x_age)
            male_mask, female_mask = y_gender_sub == 0, y_gender_sub == 1
            if np.any(male_mask) and np.any(female_mask):
                mu_male = np.mean(z_means[male_mask], axis=0)
                mu_female = np.mean(z_means[female_mask], axis=0)
                vec_gender = mu_female - mu_male
                vec_age = orthogonalize_against(vec_age, vec_gender, preserve_magnitude=True)
            model_cache[model_name] = (vae, vec_age, resolution)
        return model_cache[model_name]

    # Load default image on startup (may be outside filtered age range)
    dataset_path = serve_state["dataset_path"]
    default_image_path = Path(dataset_path) / DEFAULT_IMAGE_FILENAME
    if default_image_path.exists():
        vae, vec_age, res = get_vae_and_vec(default_model)
        initial_img = load_single_image(str(default_image_path), target_size=res)
        initial_output = age_person(vae, initial_img, vec_age, intensity=0.0)
        initial_display = _img_for_display(initial_output)
        initial_path = str(default_image_path)
    else:
        initial_display = None
        initial_img = None
        initial_path = None

    def _filename_text(path: str | None) -> str:
        return f"*{Path(path).name}*" if path else ""

    with gr.Blocks(title="Aging VAE") as demo:
        # Store model name (string), not VAE (cannot be deepcopied)
        model_state = gr.State(value=default_model)
        current_image = gr.State(value=initial_img)
        current_path = gr.State(value=initial_path)

        gr.Markdown("# Aging VAE — Face Age Manipulation")
        gr.Markdown(
            "Click **Random Image** to select a face, then **Animate** to watch them age and rejuvenate."
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(model_choices.keys()),
                value=default_model,
                label="Model",
            )
            random_btn = gr.Button("Random Image", variant="primary")
            animate_btn = gr.Button("Animate", variant="secondary")

        output_img = gr.Image(
            label="Output",
            type="numpy",
            elem_id="output-image",
            height=DISPLAY_SIZE + CONTAINER_PADDING,
            width=DISPLAY_SIZE + CONTAINER_PADDING,
            value=initial_display,
        )
        age_slider = gr.Slider(
            minimum=0,
            maximum=ANIMATION_STEPS,
            step=1,
            value=0,
            label="Age intensity (slide to preview, 0=neutral)",
        )
        resolution_label = gr.Markdown(
            value=_filename_text(initial_path) if initial_display is not None else "",
            visible=initial_display is not None,
        )

        def on_random(model_name):
            """Pick random path, load that one image, display at neutral age."""
            path_idx = np.random.randint(0, len(paths))
            path = paths[path_idx]
            vae, vec_age, resolution = get_vae_and_vec(model_name)
            img = load_single_image(path, target_size=resolution)
            # Show at neutral (no aging)
            output = age_person(vae, img, vec_age, intensity=0.0)
            return (
                _img_for_display(output),
                img,
                path,
                gr.update(visible=True, value=_filename_text(path)),
                gr.update(value=0),
            )

        def on_animate(model_name, current_image):
            """Generator that yields frames: 0 -> 3.5 (older) -> 0 (back to start)."""
            if current_image is None:
                yield None, 0
                return
            vae, vec_age, _ = get_vae_and_vec(model_name)
            # Phase 1: 0 -> MAX_AGE_INTENSITY (aging forward)
            for i in range(ANIMATION_STEPS + 1):
                intensity = (i / ANIMATION_STEPS) * MAX_AGE_INTENSITY
                aged = age_person(vae, current_image, vec_age, intensity=intensity)
                yield _img_for_display(aged), 0
                time.sleep(ANIMATION_FRAME_DELAY)
            # Phase 2: MAX_AGE_INTENSITY -> 0 (aging back to neutral)
            for i in range(ANIMATION_STEPS + 1):
                intensity = MAX_AGE_INTENSITY - (i / ANIMATION_STEPS) * MAX_AGE_INTENSITY
                aged = age_person(vae, current_image, vec_age, intensity=intensity)
                yield _img_for_display(aged), 0
                time.sleep(ANIMATION_FRAME_DELAY)

        def on_slider(model_name, current_image, slider_value):
            """Render aged image for the given slider step (0..ANIMATION_STEPS)."""
            if current_image is None:
                return None
            vae, vec_age, _ = get_vae_and_vec(model_name)
            intensity = (slider_value / ANIMATION_STEPS) * MAX_AGE_INTENSITY
            aged = age_person(vae, current_image, vec_age, intensity=intensity)
            return _img_for_display(aged)

        def on_model_change(model_name, current_path, current_image):
            """When model changes, reload image at new resolution and re-render."""
            if current_path is None:
                return model_name, None, None, gr.update(visible=False), 0
            vae, vec_age, resolution = get_vae_and_vec(model_name)
            img = load_single_image(current_path, target_size=resolution)
            output = age_person(vae, img, vec_age, intensity=0.0)
            return (
                model_name,
                _img_for_display(output),
                img,
                gr.update(visible=True, value=_filename_text(current_path)),
                0,
            )

        random_btn.click(
            fn=on_random,
            inputs=[model_state],
            outputs=[output_img, current_image, current_path, resolution_label, age_slider],
            api_name=False,
        )

        animate_btn.click(
            fn=on_animate,
            inputs=[model_state, current_image],
            outputs=[output_img, age_slider],
            api_name=False,
        )

        age_slider.change(
            fn=on_slider,
            inputs=[model_state, current_image, age_slider],
            outputs=[output_img],
            api_name=False,
        )

        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown, current_path],
            outputs=[model_state, output_img, current_image, resolution_label, age_slider],
            api_name=False,
        )

        gr.Markdown("---\n*Powered by Old Zin Software Corp*")

    return demo
