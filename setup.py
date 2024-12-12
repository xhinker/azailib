from setuptools import setup,find_packages

setup(
    name               = 'azailib'
    , version          = '1.241212.1'
    , license          = 'Apache License'
    , author           = "Andrew Zhu"
    , author_email     = 'xhinker@hotmail.com'
    , packages         = find_packages(where='src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/xhinker/azailib'
    , keywords         = 'Andrew Zhu AI tool lib'
    , install_requires = [
        'optimum-quanto'
        , 'torch'
        , 'torchvision'
        , 'torchaudio'
        , 'sentencepiece'
        , 'accelerate'
        , 'peft'
        , 'transformers'
        , 'diffusers'
        , 'lark'
        , 'protobuf'
        , 'ipykernel'
        , 'ipywidgets'
        , 'safetensors'
        , 'groundingdino-py'
        , 'rembg'
    ]
    , include_package_data=True
)