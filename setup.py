
from setuptools import setup


setup(
	name             		= 'worklab',
	version          		= '0.1.0',
	description      		= 'Basic scripts for worklab devices',
	author           		= 'Rick de Klerk',
	author_email     		= 'r.de.klerk@umcg.nl',
	url              		= '..',
	download_url     		= '..',
	packages         		= ['worklab'],
	package_data     		= {'worklab' : ['example_data/*.*'] },
	include_package_data 	= True,
	long_description 		= '..',
	license 				= 'GNU GPLv3',
	keywords         		= ['biomechanics', 'physiology'],
	classifiers      		= [],
	install_requires 		= ["numpy", "scipy", "pandas", "matplotlib"]
)