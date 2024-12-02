# azailib
<a href="https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373" target="_blank"><img src="https://m.media-amazon.com/images/I/81qJBJlgGEL._SL1500_.jpg" alt="Using Stable Diffusion with Python" height="256px" align="right"></a>

AI models like Stable Diffusion require image pre-process, generate canny image etc. this repo is for holding these common tool in one repo. 

## Install 

`pip install -U git+https://github.com/xhinker/azailib.git`

## Use it

A sample to use azailib to load and use 8-bit flux.1 pipe
```py
import torch
from azailib.sd_pipe_loaders import load_flux1_8bit_pipe

model_path  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
pipe        = load_flux1_8bit_pipe(
    checkpoint_path_or_id   = model_path
    , pipe_device           = "cuda:0"
)

#%%
# generate image
prompt = "4k, best quality, beautiful 20 years girl, show hands and fingers"
image = pipe(
    prompt                  = prompt
    , guidance_scale        = 3.5 
    , num_inference_steps   = 24
    , height                = 1024
    , width                 = 1024
    , generator             = torch.Generator('cuda:0').manual_seed(5)
).images[0]

display(image)
```

That is it.