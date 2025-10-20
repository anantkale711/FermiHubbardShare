from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(["*.pyx"], language_level = 3, annotate=True),
#    ext_modules=cythonize("Lanczos.pyx", language_level = 3, annotate=True),
    # ext_modules=cythonize("hop_op_FH.pyx", language_level = 3),
#     ext_modules=cythonize("corr_funcs.pyx", language_level = 3),
#     ext_modules=cythonize("heisenberg.pyx", language_level = 3),
#     ext_modules=cythonize("test.pyx", language_level = 3),
    include_dirs=[numpy.get_include()],
    
)