#! /usr/bin/env python
#
# Authors: Joseph Knox josephk@alleninstitute.org
# License:

import os
import shutil

from distutils.command.clean import clean as Clean

try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

DISTNAME = 'flexible-iterative-hierarchical-clustering'
DESCRIPTION = '0.1.0'
with open('README.rst') as f:
    LONG_DESCRIPTON = f.read()
MAINTAINER = 'Joseph Knox'
MAINTAINER_EMAIL = 'josephk@alleninstitute.org'
URL = 'https://github.com/jknox13/iterative-heirarchical-clustering'
LICENSE = 'new BSD'
VERSION = '0.0.1'

# NOTE : verbatim from scikit-learn/scikit-learn/setup.py
# Custom clean command to remove build artifacts

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

# Use cython if available
try:
    from cython.build import cythonize
except ImportError:
    USE_CYTHON = False
    ext = '.pyx'
else:
    USE_CYTHON = True
    ext = '.c'

extensions = [Extension("*", ["*"+ext])]

if USE_CYTHON:
    from cython.build import cythonize
    extensions = cythonize(extensions)


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    long_description=LONG_DESCRIPTON,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Unix',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 ],
                    cmdclass=cmdclass,
                    ext_modules=extensions,
                    setup_requires=['pytest-runner'])

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
