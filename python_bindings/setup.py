from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = []

extension = CppExtension(
    'simba_cpp', ['python_bindings.cpp', '../Simba.cpp', '../defs.cpp'],
    extra_compile_args={'cxx': ['-O2']})
ext_modules.append(extension)

setup(
    name='simba_cpp',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
