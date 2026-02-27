import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_filtered_utkface_paths(dataset_path, min_age=18, max_age=80):
    """Return (paths, ages, genders) without loading images. For lazy/on-demand loading."""
    paths = []
    ages = []
    genders = []
    file_list = list(Path(dataset_path).glob("*.jpg"))
    for path in file_list:
        try:
            parts = path.name.split("_")
            if len(parts) < 3:
                continue
            age = int(parts[0])
            gender = int(parts[1])
            if min_age <= age <= max_age:
                paths.append(str(path))
                ages.append(age)
                genders.append(gender)
        except (ValueError, IndexError):
            continue
    return paths, np.array(ages), np.array(genders)


def load_single_image(path: str, target_size=(64, 64)) -> np.ndarray:
    """Load and preprocess a single image for the VAE."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img.astype("float32") / 255.0


def load_images_by_paths(paths: list[str], target_size=(64, 64)) -> np.ndarray:
    """Load multiple images by path. Used for age vector computation."""
    print(f"Loading images of size {target_size}")
    images = []
    for p in paths:
        img = load_single_image(p, target_size)
        images.append(img)
    return np.array(images)


def load_filtered_utkface(dataset_path, min_age=18, max_age=80, target_size=(64, 64)):
    images = []
    ages = []
    genders = []
    
    file_paths = list(Path(dataset_path).glob("*.jpg"))
    print(f"Filtering {len(file_paths)} images for ages {min_age}-{max_age}...")

    for path in tqdm(file_paths):
        try:
            # Filename: [age]_[gender]_[race]_[date].jpg
            parts = path.name.split('_')
            
            # Basic validation of the filename structure
            if len(parts) < 3:
                continue
                
            age = int(parts[0])
            gender = int(parts[1]) # 0: Male, 1: Female
            
            if min_age <= age <= max_age:
                # Load image
                img = cv2.imread(str(path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                
                # Normalize and ensure float32 for Apple Silicon GPU
                img = img.astype("float32") / 255.0

                images.append(img)
                ages.append(age)
                genders.append(gender)
        except (ValueError, IndexError):
            # Skip files with malformed names
            continue

    return np.array(images), np.array(ages), np.array(genders)