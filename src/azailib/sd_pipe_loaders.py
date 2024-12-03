'''
This file include function to help load Diffusion model pipe.
'''
import torch
from diffusers import (
    FluxPipeline
    , FluxFillPipeline
    , FluxTransformer2DModel
    , FluxImg2ImgPipeline
)

from torchao.quantization import (
    quantize_
    , int8_weight_only
)

################################################################################
# Flux.1 
################################################################################

def load_flux1_8bit_pipe(
    checkpoint_path_or_id:str
    , transformer_path_or_id:str    = None
    , pipe_gpu_id:int               = 0
):  
    '''
    Initializes a Flux Pipeline with an 8-bit quantized transformer model.
    
    Args:
        checkpoint_path_or_id (str): Path or ID of the pipeline checkpoint.
        transformer_path_or_id (str, optional): Transformer's path or ID. Defaults to None, which uses the checkpoint's transformer.
        pipe_gpu_id (int, optional): Device for the pipeline. Defaults to 0.

    Returns:
        FluxPipeline: The initialized pipeline with the quantized transformer.

    Raises:
        ValueError: If `checkpoint_path_or_id` is invalid.

    Examples:
        >>> pipe = load_flux1_8bit_pipe("path/to/checkpoint")
        >>> # Using a custom transformer
        >>> pipe = load_flux1_8bit_pipe("checkpoint/id", transformer_path_or_id="transformer/path")
    '''
    transformer_path_or_id = checkpoint_path_or_id if transformer_path_or_id is None else transformer_path_or_id
    quantize_device = f"cuda:{pipe_gpu_id}"
    
    transformer = FluxTransformer2DModel.from_pretrained(
        transformer_path_or_id
        , subfolder = "transformer"
        , torch_dtype = torch.bfloat16
    )
    quantize_(
        transformer
        , int8_weight_only() 
        , device = quantize_device      # quantize using GPU to accelerate the speed
    )
    pipe = FluxPipeline.from_pretrained(
        checkpoint_path_or_id
        , transformer = transformer
        , torch_dtype=torch.bfloat16
    )
    
    pipe.enable_model_cpu_offload(gpu_id = pipe_gpu_id)
    return pipe

def load_flux1_img2img_8bit_pipe(
    checkpoint_path_or_id:str
    , transformer_path_or_id:str    = None
    , pipe_gpu_id:int               = 0
):  
    '''
    Initializes a Flux Image 2 Image Pipeline with an 8-bit quantized transformer model.
    
    Args:
        checkpoint_path_or_id (str): Path or ID of the pipeline checkpoint.
        transformer_path_or_id (str, optional): Transformer's path or ID. Defaults to None, which uses the checkpoint's transformer.
        pipe_gpu_id (int, optional): Device for the pipeline. Defaults to 0.

    Returns:
        FluxPipeline: The initialized pipeline with the quantized transformer.

    Raises:
        ValueError: If `checkpoint_path_or_id` is invalid.

    Examples:
        >>> pipe = load_flux1_img2img_8bit_pipe("path/to/checkpoint")
        >>> # Using a custom transformer
        >>> pipe = load_flux1_img2img_8bit_pipe("checkpoint/id", transformer_path_or_id="transformer/path")
    '''
    transformer_path_or_id = checkpoint_path_or_id if transformer_path_or_id is None else transformer_path_or_id
    quantize_device = f"cuda:{pipe_gpu_id}"
    
    transformer = FluxTransformer2DModel.from_pretrained(
        transformer_path_or_id
        , subfolder = "transformer"
        , torch_dtype = torch.bfloat16
    )
    quantize_(
        transformer
        , int8_weight_only() 
        , device = quantize_device      # quantize using GPU to accelerate the speed
    )
    pipe = FluxImg2ImgPipeline.from_pretrained(
        checkpoint_path_or_id
        , transformer = transformer
        , torch_dtype=torch.bfloat16
    )
    
    pipe.enable_model_cpu_offload(gpu_id = pipe_gpu_id)
    return pipe

def load_flux1_fill_8bit_pipe(
    checkpoint_path_or_id:str
    , pipe_gpu_id:int               = 0
):  
    '''
    Initializes a Flux Fill Pipeline with an 8-bit quantized transformer model. Fill model can not use custom transformer weights
    
    Args:
        checkpoint_path_or_id (str): Path or ID of the pipeline checkpoint.
        pipe_gpu_id (int, optional): Device for the pipeline. Defaults to 0.

    Returns:
        FluxPipeline: The initialized pipeline with the quantized transformer.

    Raises:
        ValueError: If `checkpoint_path_or_id` is invalid.

    Examples:
        >>> pipe = load_flux1_fill_8bit_pipe("path/to/FLUX.1-Fill-dev")
    '''
    quantize_device = f"cuda:{pipe_gpu_id}"
    
    transformer = FluxTransformer2DModel.from_pretrained(
        checkpoint_path_or_id
        , subfolder     = "transformer"
        , torch_dtype   = torch.bfloat16
    )
    quantize_(
        transformer
        , int8_weight_only() 
        , device = quantize_device # quantize using GPU to accelerate the speed
    )
    pipe = FluxFillPipeline.from_pretrained(
        checkpoint_path_or_id
        , transformer = transformer
        , torch_dtype = torch.bfloat16
    )
    pipe.enable_model_cpu_offload(gpu_id = pipe_gpu_id)
    return pipe
