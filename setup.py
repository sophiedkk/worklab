import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
	name             		= 'worklab',
	version          		= '1.7.2',
	description      		= 'Basic scripts for worklab devices',
	author           		= 'Rick de Klerk, Thomas Rietveld, Rowie Janssen',
	author_email     		= 'r.de.klerk@pl.hanze.nl, t.rietveld@lboro.ac.uk, r.j.f.janssen@umcg.nl',
	url              		= 'https://gitlab.com/Rickdkk/worklab',
	download_url     		= 'https://gitlab.com/Rickdkk/worklab',
	packages         		= ['worklab'],
	package_data     		= {},
	include_package_data 	= True,
	long_description 		= read("README.rst"),
	license 				= 'GNU GPLv3',
	keywords         		= ['wheelchair biomechanics', 'ergometry', 'physiology'],
	classifiers      		= ["Programming Language :: Python",
							   "Intended Audience :: Science/Research",
							   "Operating System :: OS Independent"],
	install_requires 		= ["numpy", "scipy>=1.2.0", "pandas", "matplotlib", "xlrd"]
)
