from setuptools import setup, find_packages

setup(
    name='nplm',
    version='0.1',
    packages=find_packages(include=["nplm", "nplm.*"]),
    install_requires=[
        "numpy",
        "torch==2.1",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "tqdm",
        "wandb",
    ],
)