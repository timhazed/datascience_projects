import os
import json
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

class SAMSegmentationApp:
    def __init__(self, root_dir, sam_checkpoint):
        self.root_dir = root_dir
        self.sam_checkpoint = sam_checkpoint
        self.sam_model = self.load_sam_model()

    def load_sam_model(self):
        """ Load the SAM model and return it """
        # Load the huge SAM v2 model from the checkpoint
        sam = sam_model_registry.get("vit_h")(checkpoint=self.sam_checkpoint)
        predictor = SamPredictor(sam)
        return predictor

    def process_subdirectories(self, point_cloud_filename = "dsm.json", image_filename = "google.jpg"):
        """ Process all subdirectories containing point cloud and image files"""

        # Traverse the root directory for subdirectories containing point cloud and image files
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                dsm_path = os.path.join(subdir_path, point_cloud_filename)
                image_path = os.path.join(subdir_path, image_filename)

                if os.path.exists(dsm_path) and os.path.exists(image_path):
                    print(f"Processing {subdir}...")
                    self.process_directory(subdir_path, image_path, dsm_path)

    def process_directory(self, subdir_path, image_path, dsm_path, min_confdence_score = 0.5):
        """ Process subdirectory by segmenting image and generating mask files"""

        # Load aerial image
        image = cv2.imread(image_path)

        # Load point cloud data (dsm.json)
        with open(dsm_path, 'r') as f:
            point_cloud_data = json.load(f)

        # Perform segmentation using SAM v2
        print("\tSegmenting image")
        masks, iou_predictions, _ = self.segment_image(image)

        # Generate best mask and save it
        print("\tCalculating best mask")
        best_mask, best_mask_index, confidence_score = self.select_best_mask(masks, iou_predictions)
        self.save_masks(subdir_path, best_mask, "best_segmentation_mask.png")

        # Generate blended mask and save it
        print("\tCalculating blended mask")
        blended_mask = self.select_and_blend_roof_masks(masks, best_mask, best_mask_index, iou_predictions)
        self.save_masks(subdir_path, blended_mask, "blended_segmentation_mask.png")

        # Generate best mask with point cloud and save it
        print("\tCalculating best mask with point cloud")

        # If we have low confidence in the best mask use the blended mask
        if (confidence_score < min_confdence_score):
            best_with_point_cloud_mask = self.select_mask_with_point_cloud(blended_mask, point_cloud_data)
        else:
            best_with_point_cloud_mask = self.select_mask_with_point_cloud(best_mask, point_cloud_data)

        self.save_masks(subdir_path, best_with_point_cloud_mask, "segmentation_with_point_cloud_mask.png")

        self.render_masks_and_image(image, best_mask, blended_mask, best_with_point_cloud_mask, subdir_path, confidence_score)

    def segment_image(self, image):
        """ Segment the image and return masks, iou_predictions and low resolution masks"""        

        # Prepare image for SAM v2
        self.sam_model.set_image(image)

        # Predict the masks and Intersection over Union (IoU scores)
        masks, iou_predictions, low_res_masks = self.sam_model.predict()
        return masks, iou_predictions, low_res_masks

    def save_masks(self, subdir_path, mask, output_name):
        """ Save the generated mask """   

        # Convert the mask to uint8 format (0 or 255 for binary image)
        mask_to_save = (mask * 255).astype(np.uint8)

        # Save the mask to a file
        mask_path = os.path.join(subdir_path, output_name)
        cv2.imwrite(mask_path, mask_to_save)

        print("\tSaved segmentation mask to {}".format(mask_path))

    def select_best_mask(self, masks, iou_predictions=None):
        """ Generate the best mask """   

        mask_to_save = masks
        best_mask_index = 0
        confidence_score = None

        # If multiple masks are returned, select the best mask
        if len(masks.shape) > 2:
            # If IoU or confidence scores are available, select the best mask or pick the first one
            if iou_predictions is not None:
                best_mask_index = np.argmax(iou_predictions)  # Select the mask with the highest IoU score
                confidence_score = iou_predictions[best_mask_index]
        
        mask_to_save = masks[best_mask_index]

        return mask_to_save, best_mask_index, confidence_score

    def select_and_blend_roof_masks(self, masks, best_mask, best_mask_index, iou_predictions, blend_threshold=0.9):
        """ Generate the blended mask """   
     
        # Initialize the blended mask with the best mask
        blended_mask = best_mask.copy()

        # Blend in other masks that have an IoU score within the blend threshold
        for i in range(len(iou_predictions)):
            if i != best_mask_index and iou_predictions[i] >= blend_threshold * iou_predictions[best_mask_index]:
                # Perform element-wise maximum to blend multiple masks
                blended_mask = np.maximum(blended_mask, masks[i])

        return blended_mask

    def select_mask_with_point_cloud(self, best_mask, point_cloud):
        """ Refines the best mask using point cloud height data with dynamic thresholding """   
 
        # Convert point_cloud from list of lists to a NumPy array
        point_cloud = np.array(point_cloud)

        # Initialize the new mask to have the same shape as best_mask
        refined_mask = np.zeros(best_mask.shape, dtype=np.uint8)
        
        # Extract z-values (height) from the point cloud
        z_values = point_cloud[:, 2]  
        
        # Calculate height statistics (mean and standard deviation of z-values)
        mean_height = np.mean(z_values)
        std_height = np.std(z_values)
        height_threshold = mean_height + std_height  # Set a threshold based on 1 standard deviation above the mean
        
        # Refine mask based on height threshold
        for point in point_cloud:
            x, y, z = point[:3]
            # Ensure the coordinates are within the image bounds
            if 0 <= x < best_mask.shape[1] and 0 <= y < best_mask.shape[0] and z <= height_threshold:
                refined_mask[int(y), int(x)] = 255  # Set the mask pixel to white for valid points
        
        # Ensure both masks are of type uint8
        best_mask = best_mask.astype(np.uint8)

        # Combine the refined mask with the best mask using bitwise_or
        combined_mask = cv2.bitwise_or(best_mask, refined_mask)
        
        return combined_mask

    def render_masks_and_image(self, source_image, best_mask, blended_mask, point_cloud_mask, subdir_path, confidence_score):
        """ Visualize the best mask, blended mask, best mask using point cloud, and the original aerial image. """   
    
        # Create a figure with four subplots (4 panes)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Display the best mask
        axes[0].imshow(best_mask, cmap='gray')
        axes[0].set_title('Best Mask confidence: {:.2f}'.format(confidence_score), fontsize=12)
        axes[0].axis('off')

        # Display the blended mask
        axes[1].imshow(blended_mask, cmap='gray')
        axes[1].set_title('Blended Mask', fontsize=12)
        axes[1].axis('off')

        # Display the best mask using point cloud
        axes[2].imshow(point_cloud_mask, cmap='gray')
        axes[2].set_title('Mask using point cloud data', fontsize=12)
        axes[2].axis('off')

        # Display the source aerial image
        axes[3].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Source Image', fontsize=12)
        axes[3].axis('off')

        # Set the overall window and figure title
        fig.canvas.manager.set_window_title("Mask and Image Visualizer")
        fig.suptitle(f"Processed subdirectory: {os.path.basename(subdir_path)}", fontsize=14)

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show(block=True)