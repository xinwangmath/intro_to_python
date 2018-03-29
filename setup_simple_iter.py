from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='simple_iter',
    ext_modules = cythonize('simple_iter.pyx')
)

# Run `python setup_simple_iter.py build_ext --inplace` in command line
# build_ext option tells Cython to build the module in ext_modules
# --inplace option tells Cython to build the module in the same location (instead of in the build directory)