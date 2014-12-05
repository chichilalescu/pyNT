#! /usr/bin/env python
#######################################################################
#                                                                     #
#  Copyright 2014 Cristian C Lalescu                                  #
#                                                                     #
#  This file is part of pyNT.                                         #
#                                                                     #
#  pyNT is free software: you can redistribute it and/or modify       #
#  it under the terms of the GNU General Public License as published  #
#  by the Free Software Foundation, either version 3 of the License,  #
#  or (at your option) any later version.                             #
#                                                                     #
#  pyNT is distributed in the hope that it will be useful,            #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of     #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
#  GNU General Public License for more details.                       #
#                                                                     #
#  You should have received a copy of the GNU General Public License  #
#  along with pyNT.  If not, see <http://www.gnu.org/licenses/>       #
#                                                                     #
#######################################################################


########################################################################
#
# some global settings
#
AUTHOR = 'Cristian C Lalescu'
AUTHOR_EMAIL = ''
#
########################################################################



########################################################################
#
# define version
#
import datetime
now = datetime.datetime.now()
date_name = '{0:0>4}{1:0>2}{2:0>2}'.format(now.year, now.month, now.day)
VERSION = date_name
#
########################################################################



from setuptools import setup, Extension
setup(
        name = 'pyNT',
        version = VERSION,
        packages = ['pyNT'],
        install_requires = ['numpy>=1.8', 'sympy>=0.7.5'],

        #### package description stuff goes here
        description = 'Numerical Trajectories in Python',
        long_description = open('README.rst', 'r').read(),
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = 'GNU GPLv3',
        classifiers = [
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        )

