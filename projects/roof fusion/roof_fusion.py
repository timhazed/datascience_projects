import logging
import argparse
from preprocessing import Preprocessing
from visualization import Visualization


def main():
    """Main entry point for the 3D Roof Reconstruction."""
    
    # Argument parser to take in the directory path for the DSM and aerial image
    parser = argparse.ArgumentParser(description="3D Roof Fusion from DSM and aerial image.")
    parser.add_argument('--dir', type=str, required=True, help='Directory containing dsm.json and aerial image (google.jpg)')
    parser.add_argument('--harris', action='store_true', help='Use Harris corners algorithm, False by default so shi-tomasi are used')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting 3D Roof Reconstruction.")
    
    # Initialize the Preprocessing object with the directory
    directory = args.dir
    logging.info(f"Processing directory: {directory}")
    preprocessor = Preprocessing(directory, args.harris)

    # Process the DSM and aerial image
    image, dsm_data, corners, normals = preprocessor.process()
   
    logging.info(f"Number of corners after subsampling: {len(corners)}")

    # After processing, initialize Visualization with processed data
    visualizer = Visualization(image, dsm_data)
    
    # Visualize the aerial image with corners
    visualizer.plot_aerial_image(image, corners)

    # Visualize the 3D scene with normals
    max_z = dsm_data[:, 2].max()  # Max Z from the point cloud
    visualizer.plot_3d_with_image_and_normals(dsm_data, normals, corners, max_z)
    
if __name__ == "__main__":
    main()
