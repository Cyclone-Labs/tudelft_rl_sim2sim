"""
Setup script for quad_controller package
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Portable quadcopter controller trained with domain randomization"

setup(
    name="quad-controller",
    version="1.0.0",
    author="Domain Randomization Quadcopter Racing",
    description="A neural network controller for quadcopter racing that generalizes across platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "quad_controller": ["*.zip"],  # Include model weights
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "stable-baselines3>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "matplotlib",
            "jupyter",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
