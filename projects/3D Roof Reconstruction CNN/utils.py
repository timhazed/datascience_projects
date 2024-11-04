import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Utilities:
    def __init__(self, max_pointcloud_length, max_planes, max_perimeter_length):
        self.max_pointcloud_length = max_pointcloud_length
        self.max_planes = max_planes
        self.max_perimeter_length = max_perimeter_length

    def _normalize_and_pad_planes(self, planes_data):
        azimuths, tilts, heights, perimeters = [], [], [], []

        for i, plane in enumerate(planes_data):
            if i >= self.max_planes:
                break  # Limit to max_planes planes

            # Normalize and collect data for each feature
            azimuths.append(plane.get("azimuth", 0.0) / 360.0)
            tilts.append(plane.get("tilt", 0.0) / 90.0)
            heights.append(plane.get("height", 0.0) / 100.0)

            # Normalize and pad perimeter points
            perimeter_points = plane.get("perimeter", [])
            perimeter = [(point["x"] / 100.0, point["y"] / 100.0) for point in perimeter_points[:self.max_perimeter_length]]

            # Add padding if necessary
            if len(perimeter) < self.max_perimeter_length:
                perimeter.extend([(0.0, 0.0)] * (self.max_perimeter_length - len(perimeter)))  

            perimeters.append(perimeter)

        # Pad azimuths, tilts, heights, and perimeters to self.max_planes length
        azimuths.extend([0.0] * (self.max_planes - len(azimuths)))
        tilts.extend([0.0] * (self.max_planes - len(tilts)))
        heights.extend([0.0] * (self.max_planes - len(heights)))
        
        # Add padding for perimeters if fewer than self.max_planes
        if len(perimeters) < self.max_planes:
            perimeters.extend([[(0.0, 0.0)] * self.max_perimeter_length] * (self.max_planes - len(perimeters)))

        # Convert lists to numpy arrays
        azimuths = np.array(azimuths)
        tilts = np.array(tilts)
        heights = np.array(heights)
        perimeters = np.array(perimeters)

        return azimuths, tilts, heights, perimeters

    def _normalize_data(self, images, pointclouds, planes_data, resnet50):
        # Normalize images based on the model type
        if resnet50:
            normalized_images = resnet50_preprocess(images)
        else:
            normalized_images = efficientnet_preprocess(images)

        # Normalize point clouds (assuming standardization if required)
        normalized_pointclouds = (pointclouds - np.mean(pointclouds, axis=0)) / np.std(pointclouds, axis=0)

        # Normalize and pad labels for azimuth, tilt, height, and perimeter
        azimuths, tilts, heights, perimeters = self._normalize_and_pad_planes(planes_data)

        return normalized_images, normalized_pointclouds, azimuths, tilts, heights, perimeters

    def preprocess_pointcloud(self, dsm_path):
        """
        Preprocess point cloud data by loading, padding, or trimming to a fixed number of points.
        """
        logging.info(f"Preprocessing point cloud: {dsm_path}")

        with open(dsm_path, 'r') as f:
            pointcloud = json.load(f)
        
        # Convert point cloud to a NumPy array
        pointcloud_np = np.array(pointcloud)
        
        # Trim or pad the point cloud
        if len(pointcloud_np) > self.max_pointcloud_length:
            pointcloud_np = pointcloud_np[:self.max_pointcloud_length]
        elif len(pointcloud_np) < self.max_pointcloud_length:
            padding = np.zeros((self.max_pointcloud_length - len(pointcloud_np), pointcloud_np.shape[1]))
            pointcloud_np = np.vstack((pointcloud_np, padding))
        
        logging.info(f"Point cloud processed with shape: {pointcloud_np.shape}")
        return pointcloud_np

    def process_directory(self, directory, resnet50=True):
        images, pointclouds, azimuths, tilts, heights, perimeters = [], [], [], [], [], []

        if os.path.isdir(directory):
            image_path = os.path.join(directory, "google.jpg")
            pointcloud_path = os.path.join(directory, "dsm.json")
            planes_path = os.path.join(directory, "planes.json")

            # Load raw data
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = np.array(image)

            pointcloud = self.preprocess_pointcloud(pointcloud_path)
            with open(planes_path, 'r') as f:
                planes_data = json.load(f)

            # Normalize and pad all data
            normalized_image, normalized_pointcloud, azimuth, tilt, height, perimeter = self._normalize_data(
                image, pointcloud, planes_data, resnet50
            )

            # Collect data for batch processing
            images.append(normalized_image)
            pointclouds.append(normalized_pointcloud)
            azimuths.append(azimuth)
            tilts.append(tilt)
            heights.append(height)
            perimeters.append(perimeter)

        # Convert lists to arrays outside the `if` block and remove any singleton dimensions
        images = np.squeeze(np.array(images))
        pointclouds = np.squeeze(np.array(pointclouds))
        azimuths = np.squeeze(np.array(azimuths))
        tilts = np.squeeze(np.array(tilts))
        heights = np.squeeze(np.array(heights))
        perimeters = np.squeeze(np.array(perimeters)) 

        # Package labels together in a list
        labels = [azimuths, tilts, heights, perimeters]

        return images, pointclouds, labels
    
    def process_directories(self, root_dir, model, subset_size = None):
        images = []
        pointclouds = []
        azimuth_labels = []
        height_labels = []
        tilt_labels = []
        perimeter_labels = []
    
        subdirs = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]
        if subset_size:
            subdirs = subdirs[:subset_size]
        logging.info(f"Processing {len(subdirs)} directories")

        for subdir in subdirs:
            logging.info(f"Processing directory: {subdir}")

            image, pointcloud, label = self.process_directory(subdir, resnet50=(model=='ResNet50'))

            # If we do not get a label the skip the file
            if (label):
                images.append(image)
                pointclouds.append(pointcloud)
                azimuth_labels.append(label[0])
                height_labels.append(label[1])
                tilt_labels.append(label[2])
                perimeter_labels.append(label[3])

        # Now pass the separate label arrays to the model's training function
        labels = [azimuth_labels, height_labels, tilt_labels, perimeter_labels]
        return images, pointclouds, labels



