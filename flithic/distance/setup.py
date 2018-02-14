from __future__ import division, print_function, absolute_import
from os.path import join

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.misc_util import get_info as get_misc_info
    from numpy.distutils.system_info import get_info as get_sys_info

    config = Configuration('distance', parent_package, top_path)

    config.add_data_dir('tests')

    # _distance_wrap
    config.add_extension('_distance_wrap',
                         sources=[join('src', 'distance_wrap.c')],
                         depends=[join('src', 'distance_impl.h')],
                         include_dirs=[get_numpy_include_dirs()],
                         extra_info=get_misc_info("npymath"))

    config.add_extension('_hausdorff', sources=['_hausdorff.c'])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
