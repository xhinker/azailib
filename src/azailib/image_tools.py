'''
Tool functions for image processing
'''
from PIL import Image
from diffusers.utils import load_image


def generate_outpaint_image_mask(
    input_image: str,
    top_expand: int = 100,
    right_expand: int = 100,
    bottom_expand: int = 100,
    left_expand: int = 100
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
    
    # Calculate new dimensions with expansions
    new_width = orig_width + left_expand + right_expand
    new_height = orig_height + top_expand + bottom_expand
    
    # Create a new image for the expanded version, filled with black
    expanded_img = Image.new('RGB', (new_width, new_height), color='black')
    
    # Paste the original image into the center of the new image
    expanded_img.paste(img, (left_expand, top_expand))
    
    # Create a new image for the mask, filled with black (same size as the expanded image)
    mask_img = Image.new('L', (new_width, new_height), color='white')
    
    # Draw a white rectangle on the mask to represent the original image area
    mask_img.paste('black', (left_expand, top_expand, orig_width + left_expand, orig_height + top_expand))
    
    return expanded_img, mask_img