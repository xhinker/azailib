'''
This file include function to help load Diffusion model pipe.
'''
import torch
from diffusers import (
    FluxPipeline
    , FluxFillPipeline
    , FluxTransformer2DModel
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
    , pipe_device:str               = "cuda:0"
    , quantize_device               = None
):  
    '''
    Initializes a Flux Pipeline with an 8-bit quantized transformer model.
    
    Args:
        checkpoint_path_or_id (str): Path or ID of the pipeline checkpoint.
        transformer_path_or_id (str, optional): Transformer's path or ID. Defaults to None, which uses the checkpoint's transformer.
        pipe_device (str, optional): Device for the pipeline. Defaults to "cuda:0".
        quantize_device: Device used for quantization. Defaults to `pipe_device` if not specified.

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
    quantize_device = pipe_device if quantize_device is None else quantize_device
    
    transformer = FluxTransformer2DModel.from_pretrained(
        transformer_path_or_id
        , subfolder = "transformer"
        , torch_dtype = torch.bfloat16
    )
    quantize_(
        transformer
        , int8_weight_only() 
        , device = quantize_device # quantize using GPU to accelerate the speed
    )
    pipe = FluxPipeline.from_pretrained(
        checkpoint_path_or_id
        , transformer = transformer
        , torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    return pipe

def load_flux1_fill_8bit_pipe(
    checkpoint_path_or_id:str
    , pipe_device:str               = "cuda:0"
    , quantize_device               = None
):  
    '''
    Initializes a Flux Fill Pipeline with an 8-bit quantized transformer model. Fill model can not use custom transformer weights
    
    Args:
        checkpoint_path_or_id (str): Path or ID of the pipeline checkpoint.
        pipe_device (str, optional): Device for the pipeline. Defaults to "cuda:0".
        quantize_device: Device used for quantization. Defaults to `pipe_device` if not specified.

    Returns:
        FluxPipeline: The initialized pipeline with the quantized transformer.

    Raises:
        ValueError: If `checkpoint_path_or_id` is invalid.

    Examples:
        >>> pipe = load_flux1_fill_8bit_pipe("path/to/FLUX.1-Fill-dev")
    '''
    quantize_device = pipe_device if quantize_device is None else quantize_device
    
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
    pipe.enable_model_cpu_offload()
    return pipe