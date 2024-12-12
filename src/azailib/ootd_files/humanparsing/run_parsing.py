import torch
from typing import Union
import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from .parsing_api import onnx_inference
from PIL import Image
from diffusers.utils import load_image

class Parsing:
    def __init__(
        self
        , gpu_id: int
        , humanparsing_atr_checkpoint_path:str = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_atr.onnx"
        , humanparsing_lip_checkpoint_path:str = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_lip.onnx"
    ):
        self.gpu_id                                 = gpu_id
        torch.cuda.set_device(gpu_id)
        
        session_options                             = ort.SessionOptions()
        session_options.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))
        
        self.session = ort.InferenceSession(
            humanparsing_atr_checkpoint_path
            , sess_options  = session_options
            , providers     = ['CUDAExecutionProvider'] # ['CPUExecutionProvider']
        )
        self.lip_session = ort.InferenceSession(
            humanparsing_lip_checkpoint_path
            , sess_options  = session_options
            , providers     = ['CUDAExecutionProvider'] # ['CPUExecutionProvider']
        )
        
    def __call__(
        self
        , image_or_path:Union[Image.Image, str]
    ):
        '''
        get the parsed human body mask, the model works in (384, 512) size internally
        need to convert back to its original size before returning.
        '''
        torch.cuda.set_device(self.gpu_id)
        
        if isinstance(image_or_path, Image.Image):
            input_image = image_or_path
        elif isinstance(image_or_path, str):
            input_image = load_image(image_or_path)
        
        original_w, original_h = input_image.size
        input_image = input_image.resize((384, 512))

        parsed_image, face_mask = onnx_inference(
            self.session
            , self.lip_session
            , input_image
        )
        
        parsed_image = parsed_image.resize((original_w, original_h))
        return parsed_image, face_mask