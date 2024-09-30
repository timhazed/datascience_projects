import os
import json
import numpy as np

class PointCloudManager:
    def __init__(self, root_dir, json_file):
        self.root_dir = root_dir
        self.json_file = json_file
        self.directories = self.find_subdirectories_with_dsm()

    def find_subdirectories_with_dsm(self):
        """Find all subdirectories containing the specified JSON file and return them sorted by their leaf directory names."""
        directories = []
        try:
            for subdir, _, files in os.walk(self.root_dir):
                if self.json_file in files:
                    directories.append(subdir)
        except Exception as e:
            raise OSError(f"Error accessing directory {self.root_dir}: {str(e)}")

        if not directories:
            raise FileNotFoundError(f"No directories containing '{self.json_file}' were found in '{self.root_dir}'")

        # Sort directories by the leaf directory name
        directories.sort(key=lambda x: os.path.basename(x))

        return directories

    def load_points_from_directory(self, directory):
        """Load the point cloud from the given directory."""
        json_file_path = os.path.join(directory, self.json_file)
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File '{json_file_path}' not found.")
        with open(json_file_path, 'r') as f:
            point_cloud = json.load(f)
        return np.array(point_cloud)

    def load_point_cloud(self, index):
        """Load the point cloud from a specific directory based on the index."""
        if index < 0 or index >= len(self.directories):
            raise IndexError("Index out of bounds for available directories.")
        directory = self.directories[index]
        return self.load_points_from_directory(directory)
