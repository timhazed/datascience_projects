# Roof Fusion 3D Reconstruction Application

This repository contains the Python code for a 3D roof reconstruction application based on aerial images and Digital Surface Model (DSM) data. The application performs preprocessing, visualization, and corner detection for the roof planes and normal vectors.

## Project Structure

### Files:

- **`roof_fusion.py`**: The main entry point of the application. This file handles the processing pipeline for roof reconstruction using aerial images and DSM data.
- **`preprocessing.py`**: Contains the `Preprocessing` class which handles the loading of DSM data, height map extraction, and Harris corner detection.
- **`visualization.py`**: Contains the `Visualization` class which provides functionalities for plotting the aerial image with boundaries and rendering the point cloud in 3D.

### Main Functionalities:

- **3D Roof Reconstruction**: The pipeline combines DSM data and aerial images to detect roof corners and normals.
- **Harris Corner Detection**: Detects corners in the roof structure for further processing and visualization.
- **Visualization**: The application provides 2D and 3D visualizations of the aerial image, and roof structure.

## Dependencies

This project relies on several Python libraries. You can install the required dependencies using Conda.

```bash
conda create -n roof_fusion_env python=3.9 -y
conda activate roof_fusion_env

# Install necessary dependencies
conda install -c conda-forge numpy pandas matplotlib scikit-learn scikit-image opencv pyvista pillow jsonschema h5py -y
conda install -c defaults tensorflow keras
