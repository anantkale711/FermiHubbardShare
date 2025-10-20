Make the setup.py file something like this:

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("example.pyx")
)


-------------

To compile the file run this in terminal:

python setup.py build_ext --inplace --force

--------------------------------

