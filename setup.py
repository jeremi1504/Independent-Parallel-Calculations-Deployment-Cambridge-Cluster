from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ZScorePW",
        ["ZscorePWCython.pyx"],
        language="c++",  # Specify the use of C++
        # compiler_directives={"language_level": "3"},
        # extra_compile_args=["-std=c++11", "-B", "/home/jeremi/anaconda3/compiler_compat"],
        include_dirs=[np.get_include()],
        
    ),
]

setup(
    name='ZScorePW',
    ext_modules=cythonize(extensions),
)