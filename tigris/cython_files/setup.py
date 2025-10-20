from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("hop_op_FH.pyx"),
)