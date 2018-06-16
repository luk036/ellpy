#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for ellpy

You can install ellpy with

python setup.py install
"""
# from glob import glob
import os
import sys
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from setuptools import setup

if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

if sys.version_info[:2] < (2, 7):
    print("ellpy requires Python 2.7 or later (%d.%d detected)." %
          sys.version_info[:2])
    sys.exit(-1)

# Write the version information.
sys.path.insert(0, 'ellpy')
import release
version = release.write_versionfile()
sys.path.pop(0)

packages = ["ellpy",
            "ellpy.oracles",
            "ellpy.tests"]

# docdirbase = 'share/doc/ellpy-%s' % version
# # add basic documentation
# data = [(docdirbase, glob("*.txt"))]
# # add examples
# for d in ['.',
#           'advanced',
#           'algorithms',
#           'basic',
#           '3d_drawing',
#           'drawing',
#           'graph',
#           'javascript',
#           'jit',
#           'pygraphviz',
#           'subclass']:
#     dd = os.path.join(docdirbase, 'examples', d)
#     pp = os.path.join('examples', d)
#     data.append((dd, glob(os.path.join(pp, "*.txt"))))
#     data.append((dd, glob(os.path.join(pp, "*.py"))))
#     data.append((dd, glob(os.path.join(pp, "*.bz2"))))
#     data.append((dd, glob(os.path.join(pp, "*.gz"))))
#     data.append((dd, glob(os.path.join(pp, "*.mbox"))))
#     data.append((dd, glob(os.path.join(pp, "*.edgelist"))))
# # add js force examples
# dd = os.path.join(docdirbase, 'examples', 'javascript/force')
# pp = os.path.join('examples', 'javascript/force')
# data.append((dd, glob(os.path.join(pp, "*"))))

# add the tests
package_data = {
    'ellpy': ['tests/*.py']
}

install_requires = ['decorator>=4.1.0', 'numpy>=1.12.0']
extras_require = {'all': ['cvxpy', 'matplotlib']}

if __name__ == "__main__":

    setup(
        name=release.name.lower(),
        version=version,
        maintainer=release.maintainer,
        maintainer_email=release.maintainer_email,
        author=release.authors['luk036'][0],
        author_email=release.authors['luk036'][1],
        description=release.description,
        keywords=release.keywords,
        long_description=release.long_description,
        license=release.license,
        platforms=release.platforms,
        url=release.url,
        download_url=release.download_url,
        classifiers=release.classifiers,
        packages=packages,
#        data_files=data,
        package_data=package_data,
        install_requires=install_requires,
        extras_require=extras_require,
#        test_suite='nose.collector',
#        tests_require=['nose>=0.10.1'],
        zip_safe=False
    )
