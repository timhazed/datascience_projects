import os
import logging
from simple_model import SimpleRoofCNNModel
from efficientnet_model import EfficientNetRoofCNNModel
from utils import Utilities
from visualization import Visualization
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_planes_dataset(root_directory):
    max_planes = 0
    max_perimeter_points = 0

    # Iterate over each subdirectory in the root directory
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        
        # Check if the subdirectory contains the required files
        if os.path.isdir(subdir_path):
            files_in_subdir = set(os.listdir(subdir_path))
            required_files = {"google.jpg", "dsm.json", "planes.json"}

            # Continue only if all required files are present
            if required_files.issubset(files_in_subdir):
                planes_file_path = os.path.join(subdir_path, "planes.json")

                # Load and analyze the planes.json file
                with open(planes_file_path, 'r') as f:
                    planes_data = json.load(f)
                
                # Update the max_planes count
                num_planes = len(planes_data)
                max_planes = max(max_planes, num_planes)
                
                # Check each plane's perimeter points
                for plane in planes_data:
                    perimeter_points = plane.get("perimeter", [])
                    max_perimeter_points = max(max_perimeter_points, len(perimeter_points))
    
    print(f"Maximum number of planes in any file: {max_planes}")
    print(f"Maximum number of perimeter points for any plane: {max_perimeter_points}")

    return max_planes, max_perimeter_points


def main(root_dir, efficient, subset_size, augment, both):
    logging.info("Starting Roof CNN model training process")

    # Specify the root directory containing the child directories with google.jpg, dsm.json, and planes.json
    #max_planes, max_perimeter_points = analyze_planes_dataset(root_dir)
    max_planes = 24
    max_perimeter_points = 19
    max_pointcloud_length = 8196
    visualization = Visualization()

    models = {"ResNet50": SimpleRoofCNNModel(max_pointcloud_length=max_pointcloud_length, max_planes=max_planes, max_perimeter_points=max_perimeter_points,
                                             augment=augment)}

    if (both):
        models["EfficientNet"] = EfficientNetRoofCNNModel(max_pointcloud_length=max_pointcloud_length, max_planes=max_planes, 
                                                          max_perimeter_points=max_perimeter_points, augment=augment)
    elif (efficient):
        models = {"EfficientNet": EfficientNetRoofCNNModel(max_pointcloud_length=max_pointcloud_length, max_planes=max_planes, 
                                                           max_perimeter_points=max_perimeter_points, augment=augment)}

    utilities = Utilities(max_pointcloud_length=max_pointcloud_length, max_planes=max_planes, max_perimeter_length=max_perimeter_points)

    results = {}
    for key, model in models.items():
        images, pointclouds, labels = utilities.process_directories(root_dir, key, subset_size)
        history = model.train(images, pointclouds, labels)
        results[key] = history

    for key, history in results.items():
        # Visualize training performance
        if (augment):
            title = "{} Augmented Training and Validation Losses for All Components".format(key)
        else:
            title = "{} Non-Augmented Training and Validation Losses for All Components".format(key)
            
        visualization.plot_component_losses(history, title=title)

    logging.info("Completed processing")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Roof 3D Reconstruction Model")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subdirectories with google.jpg, dsm.json, and planes.json.")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size to limit the number of directories processed.")
    parser.add_argument('--augment', action='store_true', help='Augment the training dataset by default its False')

    # Create a mutually exclusive group for --both and --efficient
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--efficient', action='store_true', help='Run the efficientnet model by default its False. Note if this is set --both cannot be set')
    group.add_argument('--both', action='store_true', help='Run both models, note if this is set --efficientnet cannot be set')
    
    args = parser.parse_args()
    main(args.root_dir, args.efficient, args.subset_size, args.augment, args.both)
