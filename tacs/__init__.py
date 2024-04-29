"""
TACS is a parallel finite-element package for analysis and
gradient-based design optimization.
"""

import os

__all__ = []

def init_mkl():
    """Initialize the environment for MKL
       Since we use the SDL for MKL, we need to configure it at runtime.
       This is MUST be performed before any calls to MKL.
       Could be improved by performing the init from the C++ and actually checking that the threading layer is the one we want, instead of using env var.
    """
    # Define threading interface
    if not 'MKL_INTERFACE_LAYER' in os.environ:
        os.environ['MKL_INTERFACE_LAYER'] = 'LP64' # 32 bits (default), ILP64 for 64 bits
    # Define threading layer
    if not 'MKL_THREADING_LAYER' in os.environ:
        os.environ['MKL_THREADING_LAYER'] = 'TBB' # sequential computation will be performed with 1 TBB thread
    # Force number of OMP threads to 1, TBB threads are controlled from inside the C++ code
    if not 'MKL_NUM_THREADS' in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['MKL_DOMAIN_NUM_THREADS'] = 'MKL_DOMAIN_ALL=1'
    # Force number of OMP threads to remain unchanged
    if not 'MKL_DYNAMIC' in os.environ:
        os.environ['MKL_DYNAMIC'] = 'FALSE'

def get_cython_include():
    """
    Get the include directory for the Cython .pxd files in TACS
    """
    return [os.path.abspath(os.path.dirname(__file__))]


def get_include():
    """
    Get the include directory for the Cython .pxd files in TACS
    """
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_inc_dirs = [
        "src",
        "src/bpmat",
        "src/elements",
        "src/elements/dynamics",
        "src/elements/basis",
        "src/constitutive",
        "src/functions",
        "src/io",
        "extern/AMD/Include",
        "extern/UFconfig",
        "extern/metis/include",
    ]

    inc_dirs = []
    for path in rel_inc_dirs:
        inc_dirs.append(os.path.join(root_path, path))

    return inc_dirs


def get_libraries():
    """
    Get the library directories
    """
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_lib_dirs = ["lib"]
    libs = ["tacs"]
    lib_dirs = []
    for path in rel_lib_dirs:
        lib_dirs.append(os.path.join(root_path, path))

    return lib_dirs, libs


# Initialize MKL
init_mkl()
# Import pytacs modules
from . import pytacs
from . import caps2tacs
from .pytacs import pyTACS
from . import problems
from . import constraints

__all__.extend(["caps2tacs", "pytacs", "pyTACS", "problems", "constraints"])
