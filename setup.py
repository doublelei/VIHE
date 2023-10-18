from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "einops",
    "pyrender",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow",
    "pyquaternion",
    "matplotlib",
    "tensorboardX",
    "clip @ git+https://github.com/openai/CLIP.git",
]

__version__ = "0.0.1"
setup(
    name="VIHE",
    version=__version__,
    description="VIHE",
    long_description="",
    packages=['VIHE'],
    install_requires=requirements,
)