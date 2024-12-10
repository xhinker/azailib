'''
Tool functions for image processing. Functions list here should never call any functions
from Models to avoid loop calling.
'''
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.ops import box_convert
import cv2
from PIL import Image
from diffusers.utils import load_image


def resize_img(img_path: str, upscale_times: float, divisible_by: int = 1):
    """
    Resizes an image by a specified upscale factor and ensures the output is in RGB mode.

    Args:
        img_path (str): Path to the input image.
        upscale_times (float): Factor by which to upscale the image.
            - Greater than 1 for upscaling.
            - Between 0 and 1 for downscaling.
        divisible_by_8 (bool, optional): If True, adjusts the output image's 
            dimensions to be divisible by 8. Defaults to False.

    Returns:
        Image: The resized Pillow Image object in RGB mode.
    """

    # Check if file exists
    if not os.path.isfile(img_path):
        raise FileNotFoundError("The specified image file does not exist.")

    try:
        # Open the image file
        with Image.open(img_path) as img:
            # Ensure the image is in RGB mode (convert if necessary)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get the original dimensions
            orig_width, orig_height = img.size
            
            # Calculate new dimensions based on the upscale factor
            new_width = int(orig_width * upscale_times)
            new_height = int(orig_height * upscale_times)
            
            # If required, adjust dimensions to be divisible by the input divisible_by
            if divisible_by != 1:
                new_width = (new_width + divisible_by -1) // divisible_by * divisible_by  # Ceiling division to nearest multiple of 8
                new_height = (new_height + divisible_by-1) // divisible_by * divisible_by
            
            # Resize the image
            img_resized = img.resize((new_width, new_height))
            
            return img_resized
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def generate_outpaint_image_mask(
    input_image: str
    , top_expand: int = 100
    , right_expand: int = 100
    , bottom_expand: int = 100
    , left_expand: int = 100
    , original_image_scale:float = 1.0
) -> tuple:
    """
    Generate an expanded image by filling with black pixels around the original image 
    based on the provided expansion parameters, and create an outpaint mask where 
    the original image area is blacked out and the expanded areas are pure white.

    Args:
        input_image (str): URL or path to the input image.
        top_expand (int, optional): Number of pixels to expand at the top. Defaults to 100.
        right_expand (int, optional): Number of pixels to expand to the right. Defaults to 100.
        bottom_expand (int, optional): Number of pixels to expand at the bottom. Defaults to 100.
        left_expand (int, optional): Number of pixels to expand to the left. Defaults to 100.

    Returns:
        tuple: A tuple containing two PIL Image objects - the expanded image and the outpaint mask.
    """

    # Load the input image
    img = load_image(input_image)
    
    # Original image dimensions
    orig_width, orig_height = img.size
    
    # Apply scaling to the original image dimensions
    scaled_width = int(orig_width * original_image_scale)
    scaled_height = int(orig_height * original_image_scale)
    
    # Resize the original image according to the scale factor
    scaled_img = img.resize((scaled_width, scaled_height))
    
    # Calculate new dimensions with expansions
    new_width = scaled_width + left_expand + right_expand
    new_height = scaled_height + top_expand + bottom_expand
    
    # Create a new image for the expanded version, filled with black
    expanded_img = Image.new('RGB', (new_width, new_height), color='black')
    
    # Paste the original image into the center of the new image
    expanded_img.paste(scaled_img, (left_expand, top_expand))
    
    # Create a new image for the mask, filled with black (same size as the expanded image)
    mask_img = Image.new('L', (new_width, new_height), color='white')
    
    # Draw a white rectangle on the mask to represent the original image area
    mask_img.paste('black', (left_expand, top_expand, scaled_width + left_expand, scaled_height + top_expand))
    
    return expanded_img, mask_img

def get_xyxy_boxes(cxcywh_boxes, image_source:Image):
    '''
    convert the float cxcywh boxes to int xyxy boxes
    '''
    h, w, _ = image_source.shape
    xyxy_boxes_output = []
    boxes = cxcywh_boxes * torch.Tensor([w,h,w,h])
    xyxy_boxes = box_convert(
        boxes       = boxes
        , in_fmt    = "cxcywh"
        , out_fmt   = "xyxy"
    )
    for xyxy_box in xyxy_boxes:
        xyxy_box = [int(i) for i in xyxy_box]
        xyxy_boxes_output.append(xyxy_box)
    return xyxy_boxes_output

def convert_cv2_to_pil_img(image_data):
    '''
    The input image should be in BGR format
    '''
    img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return pil_img