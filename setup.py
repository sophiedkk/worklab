
from setuptools import setup


setup(
	name             		= 'worklab',
	version          		= '1.0.0',
	description      		= 'Basic scripts for worklab devices',
	author           		= 'Rick de Klerk',
	author_email     		= 'r.de.klerk@umcg.nl',
	url              		= 'https://gitlab.com/Rickdkk/worklab',
	download_url     		= 'https://gitlab.com/Rickdkk/worklab',
	packages         		= ['worklab'],
	package_data     		= {},
	include_package_data 	= True,
	long_description 		= '..',
	license 				= 'GNU GPLv3',
	keywords         		= ['wheelchair biomechanics', 'ergometry', 'physiology'],
	classifiers      		= [],
	install_requires 		= ["numpy", "scipy", "pandas", "matplotlib"]
)
