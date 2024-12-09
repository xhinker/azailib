'''
This file holds functions for image process related models.
'''
from typing import Union
import os
import PIL.Image
import torch
from diffusers.utils import load_image
import numpy as np
from .lib_utility import get_resource_path
from PIL import Image
from matplotlib import pyplot as plt
import cv2

# install sam2 from original github repo
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# `pip install groundingdino-py`
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model as dino_load_model
from groundingdino.util.inference import predict as dino_predict
from groundingdino.util.inference import annotate as dino_annotate
from groundingdino.util.inference import load_image as dino_load_image

from .image_tools import (
    convert_cv2_to_pil_img
    , get_xyxy_boxes
)

################################################################################
# GroundingDINO
################################################################################
class GroundingDinoPipeline:
    """Pipeline for Grounding DINO model predictions."""

    def __init__(
        self
        , checkpoint_path_or_id: str
        , gpu_id: int = 0
    ) -> None:
        """
        Initializes the GroundingDinoPipeline.

        Args:
            model_path: Path to the DINO model.
            gpu_id: ID of the GPU to use (default: 0).

        Raises:
            None

        Returns:
            None
        """
        # Relative path to the model Python file.
        model_py_path = "azailib/groundingdino_files/GroundingDINO_SwinT_OGC.py"
        # Resolve absolute path using get_resource_path.
        model_py_path = get_resource_path(relative_path=model_py_path)
        
        # Load DINO model with specified device (GPU).
        self.dino_model = dino_load_model(
            model_py_path
            , checkpoint_path_or_id
            , device = f"cuda:{gpu_id}"  # Set device based on provided gpu_id
        )
        
        # Store provided GPU ID (corrected assignment).
        self.gpu_id = gpu_id  # Corrected from hardcoded 0 to assigned gpu_id

    def predict(
        self
        , image_path: str
        , prompt: str
        , box_threshold: float = 0.35
        , text_threshold: float = 0.1
    ) -> tuple:
        """
        Performs prediction on the given image with specified prompt and thresholds.

        Args:
            image_path: Path to the input image.
            prompt: Caption/prompt for the image.
            box_threshold: Threshold for bounding boxes (default: 0.35).
            text_threshold: Threshold for text phrases (default: 0.1).

        Returns:
            A tuple containing:
                1. Annotated image (PIL format)
                2. Bounding boxes in XYXY format
        """        
        # Load image using dino_load_image.
        image_source, image = dino_load_image(image_path)
        
        # Perform prediction with loaded model and specified thresholds.
        boxes, logits, phrases = dino_predict(
            model               = self.dino_model
            , image             = image
            , caption           = prompt
            , box_threshold     = box_threshold
            , text_threshold    = text_threshold
            , device            = f"cuda:{self.gpu_id}"  # Use stored gpu_id for device
        )
        
        # Annotate original image source with prediction results.
        annotated_frame = dino_annotate(
            image_source    = image_source
            , boxes         = boxes
            , logits        = logits
            , phrases       = phrases
        )
        
        # Convert annotated frame to PIL image format.
        annotated_img = convert_cv2_to_pil_img(annotated_frame)
        
        # Transform bounding boxes to XYXY format.
        xyxy_boxes = get_xyxy_boxes(boxes, image_source)
        
        return annotated_img, xyxy_boxes

################################################################################
# Segment anything - SAM
################################################################################

class SAMModelPiple:
    def __init__(
        self
        , checkpoint_path_or_id:str
        , gpu_id:int = 0
    ) -> None:
        # the sam2 code will handle the config path, no need to call get_resource_path the solve the path problem.
        model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam_predictor = SAM2ImagePredictor(
            build_sam2(
                model_cfg_path
                , checkpoint_path_or_id
                , device = f"cuda:{gpu_id}"
            )
        )
        
    def show_points(
        self
        , coords
        , labels
        , ax
        , marker_size=375
    ):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(
        self    
        , box
        , ax
    ):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))        
        
    def show_mask(
        self
        , mask
        , ax
        , random_color=False
        , borders = True
    ):
        '''
        See `show_masks` as the usage example
        '''
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)
        return mask_image

    def show_masks(
        self
        , image
        , masks
        , scores
        , point_coords=None
        , box_coords=None
        , input_labels=None
        , borders=True
    ):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    
    def get_combine_masks(
        self
        , masks: list[np.ndarray]
        , margin: int = 0
    ) -> Image:
        """
        Combine a list of binary ndarray masks into a single PIL mask image with optional margin expansion.

        Args:
            masks (list[np.ndarray]): List of binary mask arrays (2D or 3D with last dim=1) of the same size.
            margin (int, optional): Expand combined mask edges by this many pixels. Defaults to 0.

        Returns:
            PIL.Image: Combined mask image with expanded edges (if applicable) and white background.
        """

        # Input validation
        if not masks:
            raise ValueError("Input list of masks is empty")
        
        # Get the shape of the first mask (assuming all masks have the same shape)
        ref_shape = masks[0].shape
        
        # Validate that all masks have the same shape
        for mask in masks:
            if mask.shape != ref_shape:
                raise ValueError("All masks must have the same shape")

        # Initialize the combined mask with zeros
        combined_mask = np.zeros(ref_shape, dtype=np.bool_)

        # Iterate over each mask, performing a logical OR to combine them
        for mask in masks:
            # Ensure mask is binary (0s and 1s) and has only one channel
            mask_binary = (mask > 0).reshape(ref_shape[:2])
            combined_mask = np.logical_or(combined_mask, mask_binary)

        # Apply margin expansion using morphological dilation (if margin > 0)
        if margin > 0:
            kernel = np.ones((margin*2+1, margin*2+1), np.uint8)
            combined_mask_expanded = cv2.dilate(combined_mask.astype(np.uint8), kernel)
        else:
            combined_mask_expanded = combined_mask

        # Set white background color
        color = np.array([1, 1, 1])

        # Expand combined mask image to hxwxc (c is 3 channels here)
        combined_mask_image = combined_mask_expanded.reshape(*ref_shape[:2], 1) * color

        # Convert to uint8 and scale values to [0, 255]
        combined_mask_image_uint8 = (combined_mask_image * 255).astype(np.uint8)

        # Convert BGR to RGB (if needed, for PIL compatibility)
        if combined_mask_image_uint8.ndim == 3:  # Color image
            combined_mask_rgb = cv2.cvtColor(combined_mask_image_uint8, cv2.COLOR_BGR2RGB)
        else:  # Grayscale image (already compatible)
            combined_mask_rgb = combined_mask_image_uint8

        # Convert cv2 image to PIL image
        pil_combined_mask = Image.fromarray(combined_mask_rgb)

        return pil_combined_mask
        
    def get_masks(
        self
        , image_or_path: Union[Image.Image, str]
        , xyxy_boxes:list
        , dilate_margin:int = 0 # in pixel unit
        , show_middle_masks = False
    ) -> Image:
        '''
        This function will return return the mask based on the given xyxy boxes.
        
        Args: 
            image_or_path (Union[Image.Image, str]): input image object or path.
            xyxy_boxes (list): the input boxes with target objects. 
            dilate_margin (int): the dilate margin that expand the mask edges.
        
        Returns:
            PIL.Image: Combined mask image with expanded edges (if applicable) and white background.
        '''
        if isinstance(image_or_path, Image.Image):
            input_img = image_or_path
        elif isinstance(image_or_path, str):
            input_img = load_image(image_or_path).convert("RGB")
        self.sam_predictor.set_image(input_img)
        
        masks_list = []
        for i,box in enumerate(xyxy_boxes):
            input_box = np.array(xyxy_boxes[i])
            # predict using box
            masks, scores, _ = self.sam_predictor.predict(
                point_coords     = None,
                point_labels     = None,
                box              = input_box[None, :],
                multimask_output =False,
            )
            if show_middle_masks:
                self.show_masks(input_img, masks, scores, box_coords=input_box)
            masks_list.append(masks[0])
        combined_mask = self.get_combine_masks(masks_list, margin = dilate_margin)
        return combined_mask