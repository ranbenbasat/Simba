from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Define the extension
ext_modules = [
    CppExtension(
        name='simba_cpp',
        sources=[
            'python_bindings.cpp',
            '../Simba.cpp', 
            '../defs.cpp'
        ],
        include_dirs=['..'],
        extra_compile_args={
            'cxx': ['-O2', '-std=c++17']
        }
    )
]

setup(
    name='simba_cpp',
    version='0.1.0',
    description='Python bindings for Simba quantization algorithm',
    author='Authors of ``Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness" (ACM SIGMETRICS 2026)',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0'
    ],
)
