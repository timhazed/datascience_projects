import matplotlib.pyplot as plt
import cv2
import pyvista as pv
import numpy as np
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Visualization:
    def __init__(self, image, dsm_data):
        self.image = image
        self.dsm_data = dsm_data

    def plot_3d_point_cloud(self):
        """Visualize the point cloud in 3D."""
        logging.info("Plotting 3D point cloud.")
        points = np.array(self.dsm_data)
        cloud = pv.PolyData(points)
        cloud.plot(point_size=5)

    def plot_aerial_image(self, image, corners):
        """Plot the aerial image with corners."""
        try:
            logging.info("Plotting aerial image.")

            # Check if the image is valid
            if image is None:
                logging.error("Image is None, cannot plot.")
                return
            
            # Convert the image to RGB for displaying using matplotlib
            plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.info(f"Image shape (after conversion): {plt_image.shape}")

            # Plot the image
            plt.imshow(plt_image)
            plt.title("Aerial Image with Corners")
            plt.axis('off') 
            
            # Ensure corners are detected properly and plot them
            if corners is not None and len(corners) > 0:
                logging.info(f"Number of corners detected: {len(corners)}")

                # Convert corners to numpy array if they are a list
                if isinstance(corners, list):
                    corners = np.array(corners)


                # Log the min and max coordinates of the corners to check their range
                min_corner_x, max_corner_x = np.min(corners[:, 0]), np.max(corners[:, 0])
                min_corner_y, max_corner_y = np.min(corners[:, 1]), np.max(corners[:, 1])
                logging.info(f"Corner X range: {min_corner_x} to {max_corner_x}")
                logging.info(f"Corner Y range: {min_corner_y} to {max_corner_y}")

                # Overlay the corners as green dots
                for corner in corners:
                    x, y = corner
                    plt.scatter(x, y, c='green', s=10)  # Green dots for the corners

            else:
                logging.warning("No corners were detected.")

            # Show the plot
            plt.show()

            logging.info("Aerial image plotted successfully.")
        
        except Exception as e:
            logging.error(f"Error while plotting aerial image: {str(e)}")
        
        # Clear the plot for the next visualization
        plt.clf()

    def plot_3d_with_image_and_normals(self, point_cloud, normals, corners, max_z):
        """Visualize the point cloud in 3D with the aerial image as the floor and projected normals."""
        logging.info("Plotting 3D point cloud with normals.")

        # Ensure corners and normals are numpy arrays
        corners = np.array(corners)
        normals = np.array(normals)  # Convert to numpy array if it's not already

        # Set up the plotter and create the aerial image as the floor
        plotter = pv.Plotter()

        min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])

        image_aspect_ratio = self.image.shape[1] / self.image.shape[0]
        plane_width = max_x - min_x
        plane_height = (max_x - min_x) / image_aspect_ratio

        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        # Create the plane with aerial image
        plane = pv.Plane(center=(center_x, center_y, np.min(point_cloud[:, 2])),
                        direction=(0, 0, 1),
                        i_size=plane_width, j_size=plane_height)

        texture = pv.numpy_to_texture(self.image)
        plotter.add_mesh(plane, texture=texture)

        # Add the point cloud (Light Blue)
        point_cloud_mesh = plotter.add_mesh(pv.PolyData(point_cloud), color="lightblue", point_size=5, label="Point Cloud")

        # State tracking for point cloud visibility
        point_cloud_visible = True

        # Projected normals in Gold
        tolerance = 1e-3  # Tolerance for checking intersections

        for i, corner in enumerate(corners):
            x_img, y_img = corner.ravel()  # Get 2D corner coordinates in the image space

            # Map the 2D coordinates (image space) to 3D coordinates (plane space)
            x_3d = min_x + (x_img / self.image.shape[1]) * (max_x - min_x)
            y_3d = min_y + (y_img / self.image.shape[0]) * (max_y - min_y)

            # Start point on the image (Z at the minimum height)
            start_point = np.array([x_3d, y_3d, np.min(point_cloud[:, 2])])

            # End point with normal projection
            end_point = start_point + normals[i]

            # Find the closest point in the point cloud for intersection check
            distances = np.linalg.norm(point_cloud[:, :2] - np.array([x_3d, y_3d]), axis=1)
            closest_idx = np.argmin(distances)
            closest_point_z = point_cloud[closest_idx, 2]

            # Check if the normal intersects with the point cloud within the tolerance
            if abs(end_point[2] - closest_point_z) <= tolerance:
                logging.info(f"Normal intersects at Z={closest_point_z} for corner {i}.")
                end_point[2] = closest_point_z  # Adjust the Z to match the point cloud intersection
                sphere_actor =plotter.add_mesh(pv.Sphere(center=end_point, radius=0.5), color="green")
            else:
                if end_point[2] > max_z:
                    end_point[2] = max_z  # Cap the Z at the max_z value
                plotter.add_mesh(pv.Sphere(center=end_point, radius=0.2), color="red")  # No intersection

            # Draw the normal as a gold line
            plotter.add_lines(np.array([start_point, end_point]), color="gold", width=2)

        logging.info(f"Point cloud has {point_cloud.shape[0]} points.")

        # Add interactive axes and view configuration
        plotter.add_axes(interactive=True)
        plotter.view_isometric()

        # Add a toggle function for the point cloud visibility
        def toggle_point_cloud():
            nonlocal point_cloud_visible
            if point_cloud_visible:
                logging.info("Toggling point cloud off.")
                plotter.remove_actor(point_cloud_mesh)
            else:
                logging.info("Toggling point cloud on.")
                plotter.add_mesh(pv.PolyData(point_cloud), color="lightblue", point_size=5)

            point_cloud_visible = not point_cloud_visible  # Toggle the state
            plotter.render()  # Re-render the plot after toggling

        # Bind the 'p' key to toggle the point cloud on and off
        plotter.add_key_event('p', toggle_point_cloud)

        # Add a title
        plotter.add_text("3D Point Cloud with Normals and Image Floor", font_size=12)

        # Show plot
        plotter.show()

        logging.info("3D point cloud with normals plotted successfully.")
























