import math
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.5.0.dev0",
    "name": "auto_gptq",
    "author": "PanQiWei",
    "description": "An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/PanQiWei/AutoGPTQ",
    "keywords": ["gptq", "quantization", "large-language-models", "transformers"],
    "platforms": ["windows", "linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
}


PYPI_RELEASE = os.environ.get("PYPI_RELEASE", None)
BUILD_CUDA_EXT = int(os.environ.get("BUILD_CUDA_EXT", "1")) == 1
if BUILD_CUDA_EXT:
    try:
        import torch
    except:
        print("Building cuda extension requires PyTorch(>=1.13.0) been installed, please install PyTorch first!")
        sys.exit(-1)

    CUDA_VERSION = None
    ROCM_VERSION = os.environ.get("ROCM_VERSION", None)
    if ROCM_VERSION and not torch.version.hip:
        print(
            f"Trying to compile auto-gptq for RoCm, but PyTorch {torch.__version__} "
            "is installed without RoCm support."
        )
        sys.exit(-1)

    if not ROCM_VERSION:
        default_cuda_version = torch.version.cuda
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if ROCM_VERSION:
        common_setup_kwargs["version"] += f"+rocm{ROCM_VERSION}"
    else:
        if not CUDA_VERSION:
            print(
                f"Trying to compile auto-gptq for CUDA, byt Pytorch {torch.__version__} "
                "is installed without CUDA support."
            )
            sys.exit(-1)

        # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
        if not PYPI_RELEASE:
            common_setup_kwargs["version"] += f"+cu{CUDA_VERSION}"

requirements = [
    "accelerate>=0.19.0",
    "datasets",
    "sentencepiece",
    "numpy",
    "rouge",
    "gekko",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.31.0",
    "peft",
    "tqdm",
]

extras_require = {"triton": ["triton==2.0.0"], "test": ["parameterized"]}

include_dirs = ["autogptq_cuda"]

additional_setup_kwargs = dict()
if BUILD_CUDA_EXT:
    from torch.utils import cpp_extension

    p = int(
        subprocess.run(
            "cat /proc/cpuinfo | grep cores | head -1", shell=True, check=True, text=True, stdout=subprocess.PIPE
        ).stdout.split(" ")[2]
    )

    subprocess.call(["python", "./autogptq_extension/qigen/generate.py", "--module", "--search", "--p", str(p)])
    if not ROCM_VERSION:
        from distutils.sysconfig import get_python_lib

        conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

        print("conda_cuda_include_dir", conda_cuda_include_dir)
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)
            print(f"appending conda cuda include dir {conda_cuda_include_dir}")
    extensions = [
        cpp_extension.CUDAExtension(
            "autogptq_cuda_64",
            [
                "autogptq_extension/cuda_64/autogptq_cuda_64.cpp",
                "autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu",
            ],
        ),
        cpp_extension.CUDAExtension(
            "autogptq_cuda_256",
            [
                "autogptq_extension/cuda_256/autogptq_cuda_256.cpp",
                "autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu",
            ],
        ),
        cpp_extension.CppExtension(
            "cQIGen",
            ["autogptq_extension/qigen/backend.cpp"],
            extra_compile_args=[
                "-O3",
                "-mavx",
                "-mavx2",
                "-mfma",
                "-march=native",
                "-ffast-math",
                "-ftree-vectorize",
                "-faligned-new",
                "-std=c++17",
                "-fopenmp",
                "-fno-signaling-nans",
                "-fno-trapping-math",
            ],
        ),
    ]

    if os.name == "nt":
        # On Windows, fix an error LNK2001: unresolved external symbol cublasHgemm bug in the compilation
        cuda_path = os.environ.get("CUDA_PATH", None)
        if cuda_path is None:
            raise ValueError(
                "The environment variable CUDA_PATH must be set to the path to the CUDA install when installing from source on Windows systems."
            )
        extra_link_args = ["-L", f"{cuda_path}/lib/x64/cublas.lib"]
    else:
        extra_link_args = []

    extensions.append(
        cpp_extension.CUDAExtension(
            "exllama_kernels",
            [
                "autogptq_extension/exllama/exllama_ext.cpp",
                "autogptq_extension/exllama/cuda_buffers.cu",
                "autogptq_extension/exllama/cuda_func/column_remap.cu",
                "autogptq_extension/exllama/cuda_func/q4_matmul.cu",
                "autogptq_extension/exllama/cuda_func/q4_matrix.cu",
            ],
            extra_link_args=extra_link_args,
        )
    )

    additional_setup_kwargs = {"ext_modules": extensions, "cmdclass": {"build_ext": cpp_extension.BuildExtension}}
common_setup_kwargs.update(additional_setup_kwargs)
setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    include_dirs=include_dirs,
    python_requires=">=3.8.0",
    **common_setup_kwargs,
)
