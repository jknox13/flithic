#!/usr/bin/env python
"""FlItHiC: A FLexible ITerative HIerarchical Clustering library"""

DOCLINES = __doc__

import os
import sys
import subprocess
import textwrap
import warnings


if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 2.7 or >= 3.5 required.")

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins


CLASSIFIERS="""\
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
with open('README.rst') as f:
    LONG_DESCRIPTON = f.read()
MAINTAINER = 'Joseph Knox'
MAINTAINER_EMAIL = 'josephk@alleninstitute.org'
URL = 'https://github.com/jknox13/iterative-heirarchical-clustering'
LICENSE = 'new BSD'
VERSION = '0.0.1'


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
    config.add_data_files(('flithic', '*.txt'))

    return config


def setup_package():

    cmdclass = {'sdist': sdist_checked}

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
        test_suite='nose.collector',
        setup_requires=build_requires,
        install_requires=build_requires,
        version=VERSION,
        python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    )

    if "--force" in sys.argv:
        sys.argv.remove('--force')

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
