#!/usr/bin/env python
#
# Copyright
#
# License: 3-clause BSD
"""FlItHiC: A FLexible ITerative HIerarchical Clustering library"""

import sys
import os
import shutil
import subprocess
from distutils.command.clean import clean as Clean


if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 2.7 or >= 3.5 required.")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Unix
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6

"""

DISTNAME = 'flithic'
DOCLINES = __doc__
with open('README.rst') as f:
    LONG_DESCRIPTON = f.read()
MAINTAINER = 'Joseph Knox'
MAINTAINER_EMAIL = 'josephk@alleninstitute.org'
URL = 'https://github.com/jknox13/iterative-heirarchical-clustering'
LICENSE = 'new BSD'
VERSION = '0.0.1'

# # We can actually import a restricted version of sklearn that
# # does not need the compiled code
# import flithic
#
# VERSION = flithic.__version__


# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# scipy __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__FLITHIC_SETUP__ = True



from distutils.command.sdist import sdist
class sdist_checked(sdist):
    """ check submodules on sdist to prevent incomplete tarballs """
    def run(self):
        check_submodules()
        sdist.run(self)


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         'flithic'],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('flithic')

    return config

# def get_numpy_status():
#     """
#     Returns a dictionary containing a boolean specifying whether NumPy
#     is up-to-date, along with the version string (empty string if
#     not installed).
#     """
#     numpy_status = {}
#     try:
#         import numpy
#         numpy_version = numpy.__version__
#         numpy_status['up_to_date'] = parse_version(
#             numpy_version) >= parse_version(NUMPY_MIN_VERSION)
#         numpy_status['version'] = numpy_version
#     except ImportError:
#         traceback.print_exc()
#         numpy_status['up_to_date'] = False
#         numpy_status['version'] = ""
#     return numpy_status

class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sklearn'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

def setup_package():

    cmdclass.update({'sdist': sdist_checked})

    # Figure out whether to add ``*_requires = ['numpy']``.
    # We don't want to do that unconditionally, because we risk updating
    # an installed numpy which fails too often.  Just if it's not installed, we
    # may give it a try.  See gh-3379.
    try:
        import numpy
    except ImportError:  # We do not have numpy installed
        build_requires = ['numpy>=1.8.2']
    else:
        # If we're building a wheel, assume there already exist numpy wheels
        # for this platform, so it is safe to add numpy to build requirements.
        # See gh-5184.
        build_requires = (['numpy>=1.8.2'] if 'bdist_wheel' in sys.argv[1:]
                          else [])
    finally:
        build_requires.append('pytest-runner')

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DOCLINES,
        long_description=LONG_DESCRIPTON,
        url=URL,
        download_url=URL,
        license=LICENSE,
        cmdclass=cmdclass,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Linux"],
        install_requires=build_requires,
        version=VERSION
    )

    if "--force" in sys.argv:
        sys.argv.remove('--force')

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scikit-learn when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata["version"] = VERSION

    else:

        # This import is here because it needs to be done before importing setup()
        # from numpy.distutils, but after the MANIFEST removing and sdist import
        # higher up in this file.
        from numpy.distutils.core import setup
        cwd = os.path.abspath(os.path.dirname(__file__))

        if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
            # Generate Cython sources, unless building from source release
            generate_cython()

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
