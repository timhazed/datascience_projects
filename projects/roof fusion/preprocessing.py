import numpy as np
import cv2
import json
from scipy.interpolate import griddata
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessing:
    def __init__(self, dir_path, harris, max_corners=2500):
        self.dir_path = dir_path
        self.dsm_file = f"{dir_path}/dsm.json"
        self.image_file = f"{dir_path}/google.jpg"
        self.aligned_dsm_file = f"{dir_path}/dsm_aligned.json"
        self.harris = harris
        self.max_corners = max_corners
        logging.info("Preprocessing initialized with directory: %s", dir_path)

    def load_dsm(self):
        """Load DSM data from the JSON file."""
        logging.info("Loading DSM data from: %s", self.dsm_file)
        with open(self.dsm_file, 'r') as f:
            dsm_data = json.load(f)
        logging.info("DSM data loaded successfully with %d points", len(dsm_data))
        return np.array(dsm_data, dtype=np.float64)  # Assuming DSM is a list of points [x, y, z]

    def load_image(self):
        """Load the aerial image."""
        logging.info("Loading aerial image from: %s", self.image_file)
        image = cv2.imread(self.image_file)
        if image is None:
            logging.error("Failed to load image from: %s", self.image_file)
            raise FileNotFoundError(f"Image file not found: {self.image_file}")
        logging.info("Aerial image loaded successfully with shape: %s", image.shape)
        return image

    def extract_height_map(self, dsm_data, subsample_factor=5):
        """Extract and interpolate the height map (z-values) into a regular 2D grid with subsampling."""
        logging.info("Extracting height map from DSM data with subsampling factor: %d", subsample_factor)
        
        # Subsample the DSM data
        dsm_data = dsm_data[::subsample_factor]  # Select every Nth point to reduce memory usage
        
        # Extract x, y, z values from the subsampled DSM data
        x = np.array([point[0] for point in dsm_data])
        y = np.array([point[1] for point in dsm_data])
        z = np.array([point[2] for point in dsm_data])
        
        logging.info("Unique x points: %d, Unique y points: %d", len(np.unique(x)), len(np.unique(y)))

        # Create a regular grid for x and y
        num_x = len(np.unique(x))
        num_y = len(np.unique(y))
        xi = np.linspace(min(x), max(x), num_x)
        yi = np.linspace(min(y), max(y), num_y)
        grid_x, grid_y = np.meshgrid(xi, yi)
        
        # Interpolate z values onto the regular grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        
        if grid_z is None:
            logging.error("Interpolation failed.")
            raise ValueError("Interpolation failed.")
        
        logging.info("Height map created successfully with shape: %s", grid_z.shape)
        return grid_z

    def detect_harris_corners(self, image, threshold_factor = 0.005):
        """Detect corners using Harris Corner Detector."""
        logging.info("Detecting corners using Harris Corner Detector.")
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Harris Corner Detection
        gray = np.float32(gray)
        harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        
        # Dilate for marking the corners
        harris_corners = cv2.dilate(harris_corners, None)
        
        # Thresholding for optimal corners
        threshold = threshold_factor * harris_corners.max()
        corners = np.argwhere(harris_corners > threshold)
        
        logging.info(f"Detected {len(corners)} corners.")
        
        return np.array(corners)  # Convert to numpy array if it isn't already
    
    def detect_shi_tomasi_corners(self, image, max_corners=10000, quality_level=0.005, min_distance=10):
        """Detect corners using Shi-Tomasi method."""
        logging.info("Detecting corners using Shi-Tomasi Corner Detector.")
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect corners using the Shi-Tomasi method
        shi_tomasi_corners = cv2.goodFeaturesToTrack(
            gray, max_corners, quality_level, min_distance
        )
        
        # Check if any corners were detected
        if shi_tomasi_corners is None:
            logging.warning("No corners detected by Shi-Tomasi.")
            return np.array([])  # Return an empty array if no corners found

        # Convert to integer format for consistency
        corners = np.int32(shi_tomasi_corners).reshape(-1, 2)  # Reshape to (N, 2)

        logging.info(f"Detected {len(corners)} corners using Shi-Tomasi.")
        
        return corners  # Return the transformed corners


    def calculate_normals_from_corners(self, corners, point_cloud, max_z):
        """Calculate normals from the given corners and the point cloud with max_z constraints."""
        logging.info(f"Calculating normals for {len(corners)} corners using the point cloud.")

        normals = []

        for corner in corners:
            x, y = corner.ravel()

            # Find the closest point in the point cloud to the corner (2D distance in XY)
            distances = np.linalg.norm(point_cloud[:, :2] - np.array([x, y]), axis=1)
            closest_idx = np.argmin(distances)

            # Use the point in the point cloud as a reference for the normal calculation
            point_cloud_z = point_cloud[closest_idx, 2]

            # Calculate a normal based on the height difference from the image plane (z=0) to the point cloud
            normal = np.array([0, 0, point_cloud_z])

            # Ensure the normal doesn't exceed max_z
            if normal[2] > max_z:
                normal[2] = max_z

            normals.append(normal)

        normals = np.array(normals)  # Convert the list of normals to a numpy array
        logging.info(f"Calculated {normals.shape[0]} normals.")

        return normals

    def process(self):
        """Process the DSM and aerial image, extract height map, and align the point cloud."""
        logging.info("Processing DSM and aerial image.")
        
        dsm_data = self.load_dsm()
        image = self.load_image()
        
        if (self.harris):
            corners = self.detect_harris_corners(image)
        else:
            corners = self.detect_shi_tomasi_corners(image)

        if len(corners) > self.max_corners:
            subsample_indices = np.random.choice(len(corners), self.max_corners, replace=False)
            corners = corners[subsample_indices]

        max_z = np.max(dsm_data[:, 2])
        normals = self.calculate_normals_from_corners(corners, dsm_data, max_z)
        
        logging.info("Processing completed successfully.")
        return image, dsm_data, corners, normals