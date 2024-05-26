# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# Ensure the necessary directories exist
import os
build_temp = 'build/temp.linux-x86_64-cpython-310'
if not os.path.exists(build_temp):
    os.makedirs(build_temp)

output_dir = 'pointnet2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set environment variables for the compiler
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

setup(
    name='pointnet2',
    py_modules=['pointnet2_modules', 'pointnet2_test', 'pointnet2_utils', 'pytorch_utils'],
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            # extra_compile_args={
            #     "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            #     "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            # },
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root)), '-Wno-deprecated-declarations'],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root)), '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_75,code=sm_75', '-std=c++17',
                    '-Xcompiler', '-fPIC'
                ]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
