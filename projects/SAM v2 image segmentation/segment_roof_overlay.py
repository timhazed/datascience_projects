import argparse
from SAMSegmentationApp import SAMSegmentationApp   
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM v2 Segmentation on Aerial Images and Point Clouds")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing subdirectories with dsm.json and google.jpg files')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to SAM v2 checkpoint file')

    args = parser.parse_args()

    app = SAMSegmentationApp(root_dir=args.root_dir, sam_checkpoint=args.sam_checkpoint)
    app.process_subdirectories()
