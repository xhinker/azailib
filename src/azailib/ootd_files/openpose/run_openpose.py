import pdb

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
from azailib.ootd_files.openpose.annotator.util import (
    ootd_resize_image
    , HWC3
)
from azailib.ootd_files.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch
import pdb
from azailib.image_tools import resize_img

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class OpenPose:
    def __init__(
        self
        , body_pose_checkpoint_path:str
        , gpu_id: int
    ):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.preprocessor = OpenposeDetector(
            body_pose_checkpoint_path = body_pose_checkpoint_path
        )

    def __call__(self, input_image):
        torch.cuda.set_device(self.gpu_id)
        
        if isinstance(input_image, Image.Image):
            input_image_pil = input_image
            input_image     = np.asarray(input_image_pil)
        elif type(input_image) == str:
            input_image_pil = Image.open(input_image)
            input_image     = np.asarray(input_image_pil)
        else:
            raise ValueError
        
        original_w, original_h = input_image_pil.size
        
        with torch.no_grad():
            input_image = HWC3(input_image)
            
            input_image = ootd_resize_image(input_image)
            H, W, C = input_image.shape
            
            # input_image = resize_img(image_or_path=input_image_pil, width=384, height=512)
            # W,H = input_image.size
            
            assert (H == 512 and W == 384), 'Incorrect input image shape'
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints       = {"pose_keypoints_2d": candidate}
            output_image    = cv2.resize(
                cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
                , (original_w, original_h)
            )

        return keypoints, output_image


if __name__ == '__main__':

    model = OpenPose()
    model('./images/bad_model.jpg')
