from setuptools import find_packages, setup

setup(
    name="lpe",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "torch>=2.2.1",
        "tqdm",
        "datasets>=2.17.0",
        "transformers",
        "huggingface-hub<=0.22.2",
        "jaxtyping",
        "fancy-einsum",
        "blobfile",
        "numpy",
        "boostedblob"
    ]
)