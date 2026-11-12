import os
import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0+PTX"

import torch.utils.cpp_extension as cpp_ext
cpp_ext._check_cuda_version = lambda compiler_name, compiler_version: None

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(current_dir, "csrc", "vllm_binding.cpp"),
    os.path.join(current_dir, "csrc", "gemv_kernel.cu"),
    os.path.join(current_dir, "csrc", "gemm_kernel.cu"),
]

extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++20",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
    "nvcc": [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-lineinfo",
        "--threads=4",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ]
}

setup(
    name="vllm_wlq",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="vllm_wlq._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    entry_points={
        "vllm.general_plugins": [
            "vllm_wlq = vllm_wlq:register"
        ]
    },
    cmdclass={"build_ext": BuildExtension},
    install_requires=[]
)