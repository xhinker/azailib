'''
Tool functions for image processing. Functions list here should never call any functions
from Models to avoid loop calling.
'''
import os
import numpy as np
from typing import Union
from matplotlib import pyplot as plt
import torch
from torchvision.ops import box_convert
import cv2
from PIL import Image, ImageDraw
from diffusers.utils import load_image

def resize_img(
    image_or_path: Union[Image.Image, str]
    , width: int
    , height: int
    , divisible_by: int = 1
):
    """
    Resizes an image by a specified upscale factor and ensures the output is in RGB mode.

    Args:
        img_path (str): Path to the input image.
        divisible_by_8 (bool, optional): If True, adjusts the output image's 
            dimensions to be divisible by 8. Defaults to False.

    Returns:
        Image: The resized Pillow Image object in RGB mode.
    """
    if isinstance(image_or_path, Image.Image):
        input_img = image_or_path
    elif isinstance(image_or_path, str):
        if not os.path.isfile(image_or_path):
            raise FileNotFoundError("The specified image file does not exist.")
        input_img = load_image(image_or_path).convert("RGB")

    try:
        img = input_img
        # Ensure the image is in RGB mode (convert if necessary)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate new dimensions based on the upscale factor
        new_width = width #int(orig_width * upscale_times)
        new_height = height #int(orig_height * upscale_times)
        
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

def scale_img(img_path: str, upscale_times: float, divisible_by: int = 1):
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


def concatenate_images_left_right(img_left: Image, img_right: Image) -> Image:
    """
    Concatenates two PIL Image objects side by side (left and right).

    Args:
        - img_left (PIL.Image): The left image.
        - img_right (PIL.Image): The right image.

    Returns:
        - PIL.Image: The concatenated image object.
    """
    try:
        # Ensure both images are in the same mode (e.g., both RGB or both RGBA)
        if img_left.mode != img_right.mode:
            raise ValueError("Both images must be in the same mode.")

        # Ensure images are the same height for seamless concatenation
        # If not, we'll resize the taller image to match the shorter one's height
        min_height = min(img_left.height, img_right.height)
        
        # Resize if necessary, maintaining aspect ratio by adjusting width proportionally
        if img_left.height != min_height:
            width_adjusted = int(img_left.width * (min_height / img_left.height))
            img_left = img_left.resize((width_adjusted, min_height))
        if img_right.height != min_height:
            width_adjusted = int(img_right.width * (min_height / img_right.height))
            img_right = img_right.resize((width_adjusted, min_height))
        
        # Concatenate images
        # The width will be the sum of both images' (possibly adjusted) widths; height is the minimum found
        concatenated_img = Image.new(img_left.mode, (img_left.width + img_right.width, min_height))
        concatenated_img.paste(img_left, (0, 0))  # Paste left image at (0,0)
        concatenated_img.paste(img_right, (img_left.width, 0))  # Paste right image to the right of the left one
        
        return concatenated_img
    
    except ValueError as ve:
        print(f"Value Error: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def extend_mask_left(mask_image):
    """
    Extends a given mask image to the left by mirroring its dimensions 
    with an all-black extension of equal size.

    Args:
        mask_image (PIL.Image): The input mask image.

    Returns:
        PIL.Image: The extended image with the original on the right.
    """

    # Get dimensions of the original mask image
    original_width, original_height = mask_image.size
    
    # Create a new image that is twice the width, filled with black
    new_width = original_width * 2
    new_image = Image.new('RGB', (new_width, original_height), color=(0, 0, 0))
    
    # Paste the original mask on the right half of the new image
    new_image.paste(mask_image, (original_width, 0))
    
    return new_image


def extract_object_on_white_background(img, mask_img):
    """
    Extracts an object from an image based on a provided mask and places it on a pure white background.

    Parameters:
    - img (PIL Image): The original image.
    - mask_img (PIL Image): Mask image highlighting the target object in white.

    Returns:
    - result_img (PIL Image): The extracted object on a white background, same size as the input image, in RGB mode.
    """

    # Ensure both images are the same size
    assert img.size == mask_img.size, "Input image and mask must be the same size."

    # Convert images to RGBA to allow for transparent backgrounds if needed in future adaptations
    img = img.convert('RGBA')
    mask_img = mask_img.convert('L')  # Grayscale for simpler thresholding

    # Threshold the mask to ensure only fully white pixels (255) are considered part of the object
    # You can adjust this threshold if your mask isn't purely binary
    mask_thresholded = np.array(mask_img) > 254  # Very close to white, allows for minor deviations

    # Create a white background image
    white_bg_img = Image.new('RGBA', img.size, (255, 255, 255, 255))

    # Convert PIL images to numpy arrays for easier manipulation
    img_array = np.array(img)
    white_bg_array = np.array(white_bg_img)

    # Use the mask to select pixels from the original image and place them on the white background
    white_bg_array[mask_thresholded] = img_array[mask_thresholded]

    # Convert back to PIL Image but in RGB mode
    result_img = Image.fromarray(white_bg_array[:, :, :3], 'RGB')  # Select only RGB channels, ignore Alpha

    return result_img


def extract_objects_using_xyxy_boxes(img, boxes):
    """
    Extracts objects from an image based on provided XYXY bounding boxes, 
    places them centered in a new image of the same size as the original, 
    with a white background.

    Parameters:
    - img (PIL Image): The original image.
    - boxes (list of lists): List of XYXY boxes [[x_min, y_min, x_max, y_max], ...] defining the objects to extract.

    Returns:
    - result_img (PIL Image): An RGB image, same size as the input, with extracted objects centered on a white background.
    """

    # Convert image to RGB mode for consistency
    img = img.convert('RGB')
    original_width, original_height = img.size

    # Create a new white background image of the same size as the original
    result_img = Image.new('RGB', (original_width, original_height), (255, 255, 255))

    # Calculate the total width required to place all objects side by side with a small gap
    obj_gap = 10  # Gap between objects
    total_objs_width = sum([(box[2] - box[0]) for box in boxes]) + (len(boxes) - 1) * obj_gap

    # Determine the x-coordinate to center the objects horizontally
    center_x = (original_width - total_objs_width) // 2
    current_x = center_x

    # Paste each object into the new image
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        obj = img.crop((x_min, y_min, x_max, y_max))
        
        # Calculate the y-coordinate to roughly center the object vertically
        # Assuming objects have varying heights, this might not perfectly center all objects
        # but will give a balanced look for most cases
        obj_height = obj.height
        center_y = (original_height - obj_height) // 2
        
        # Paste the object at the calculated position
        result_img.paste(obj, (current_x, center_y))
        
        # Update the current x-position for the next object
        current_x += obj.width + obj_gap

    return result_img