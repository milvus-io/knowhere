from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_py import build_py
import os

KNOWHERE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAME = "knowhere"
VERSION = os.getenv("VERSION")
if not VERSION:
    VERSION = "1.0.0"


class get_numpy_include(object):
    def __str__(self):
        import numpy as np

        return np.get_include()


class CustomBuildPy(build_py):
    """Run build_ext before build_py to compile swig code."""

    def run(self):
        self.run_command("build_ext")
        return build_py.run(self)


def CheckCUDA():
    lib_path = os.path.join(KNOWHERE_ROOT, "cmake_build/knowhere/libknowhere.so")
    x = os.popen("ldd " + lib_path).read()
    if x.find("libcudart.so") != -1:
        return True
    return False


is_cuda_ver = CheckCUDA()

DEFINE_MACROS = [
    ("FINTEGER", "int"),
    ("SWIGWORDSIZE64", "1"),
    ("SWIG_PYTHON_SILENT_MEMLEAK", "1"),
]

if is_cuda_ver:
    DEFINE_MACROS.append(("KNOWHERE_GPU_VERSION", "1"))

INCLUDE_DIRS = [
    get_numpy_include(),
    KNOWHERE_ROOT,
    os.path.join(KNOWHERE_ROOT, "knowhere"),
    os.path.join(KNOWHERE_ROOT, "thirdparty"),
    os.path.join(KNOWHERE_ROOT, "thirdparty/faiss"),
    os.path.join(KNOWHERE_ROOT, "thirdparty/any-lite/include"),
    os.path.join(KNOWHERE_ROOT, "thirdparty/nlohmann_json/include"),
    os.path.join(KNOWHERE_ROOT, "thirdparty/easyloggingpp/src"),
]

if is_cuda_ver:
    INCLUDE_DIRS.append(os.path.abspath("/usr/local/cuda/include"))


LIBRARY_DIRS = [os.path.join(KNOWHERE_ROOT, "cmake_build", "knowhere")]
EXTRA_COMPILE_ARGS = ["-fPIC", "-std=c++17"]
EXTRA_LINK_ARGS = [
    "-lknowhere",
    "-Wl,-rpath,$ORIGIN/../../../",
]

SWIG_OPTS = [
    "-c++",
    "-I" + os.path.join(KNOWHERE_ROOT, "knowhere"),
]

if is_cuda_ver:
    SWIG_OPTS.append("-DKNOWHERE_GPU_VERSION=1")

_swigknowhere = Extension(
    "knowhere._swigknowhere",
    sources=[
        os.path.join(KNOWHERE_ROOT, "python", "knowhere", "knowhere.i"),
    ],
    language="c++",
    define_macros=DEFINE_MACROS,
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    swig_opts=SWIG_OPTS,
)

setup(
    name=NAME,
    version=VERSION,
    description=(
        "A library for efficient similarity search and clustering of dense " "vectors."
    ),
    url="https://github.com/milvus-io/knowhere",
    author="milvus",
    author_email="yusheng.ma@zilliz.com",
    license="MIT",
    keywords="search nearest neighbors",
    setup_requires=["numpy"],
    packages=["knowhere"],
    data_files=[
        (
            "lib",
            [os.path.join(KNOWHERE_ROOT, "cmake_build/knowhere/libknowhere.so")],
        )
    ],
    ext_modules=[_swigknowhere],
    cmdclass={"build_py": CustomBuildPy},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
