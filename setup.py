from setuptools import setup,find_packages

setup(
    name               = 'azailib'
    , version          = '1.241209.5'
    , license          = 'Apache License'
    , author           = "Andrew Zhu"
    , author_email     = 'xhinker@hotmail.com'
    , packages         = find_packages('src')
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
    ]
    , include_package_data=True
)