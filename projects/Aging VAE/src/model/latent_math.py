import numpy as np


def orthogonalize_against(
    vec: np.ndarray, against: np.ndarray, preserve_magnitude: bool = False
) -> np.ndarray:
    """
    Remove the component of vec that aligns with 'against', producing a vector
    orthogonal to it. Used to purify the age direction by removing gender bias:
    vec_age_pure = vec_age - projection(vec_age onto vec_gender).

    Args:
        vec: The vector to orthogonalize (e.g. age direction).
        against: The vector to remove from vec (e.g. gender direction).
        preserve_magnitude: If True, rescale the result so its length matches
            the original vec. This restores the visual strength of the effect
            after orthogonalization (which otherwise shortens the vector).

    Returns:
        vec with its component along 'against' removed, optionally rescaled.
    """
    # Store original magnitude before orthogonalization (needed if rescaling)
    original_mag = np.linalg.norm(vec) if preserve_magnitude else None

    # Project vec onto 'against': proj = (vec · against / |against|²) * against
    # The scalar (vec · against) / (against · against) gives the projection length
    proj_scalar = np.dot(vec, against) / (np.dot(against, against) + 1e-8)
    result = vec - proj_scalar * against

    # Rescale to restore original magnitude so the aging effect remains strong.
    # Orthogonalization shortens the vector; without this, the same intensity
    # would produce a weaker visual change.
    if preserve_magnitude and original_mag is not None:
        new_mag = np.linalg.norm(result)
        if new_mag > 1e-8:
            result = result * (original_mag / new_mag)

    return result


def get_age_direction(vae, x_data, ages, start_age=20, end_age=70):
    # Get latent means for all images
    z_means, _, _ = vae.encoder.predict(x_data)
    
    # Calculate average position for specific age groups
    mu_start = np.mean(z_means[ages == start_age], axis=0)
    mu_end = np.mean(z_means[ages == end_age], axis=0)
    
    # The vector that moves a point from 'Young' to 'Old'
    return mu_end - mu_start

# 3. Apply the Aging Effect
def age_person(vae, image, age_vector, intensity=1.0):
    # Encode the face
    z_mean, _, _ = vae.encoder.predict(np.expand_dims(image, 0))
    
    # Move along the age vector
    z_aged = z_mean + (age_vector * intensity)
    
    # Decode back to image space
    return vae.decoder.predict(z_aged)[0]
